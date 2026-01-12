// Authors: Rohan Adla, Arrio Gonsalves, Shreyan Nalwad, Dylan Setiawan
// Date: Dec 12th 2025
// Project: A VAR-based Computational Analysis of Influenza and Weather Dynamics
// Class: 02-613 at Caregie Mellon University

package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// These functions are for the ReducedFormVAR struct and its methods
// Returns current Model Spec to see what options were selectedd
func (rf *ReducedFormVAR) Spec() ModelSpec { return rf.Model }

// Returns coefficient matrices
func (rf *ReducedFormVAR) Phi() []*mat.Dense { return rf.A }

// Returns error covariance matrix
func (rf *ReducedFormVAR) CovU() *mat.SymDense { return rf.SigmaU }

// Forecast produces multi-step ahead forecasts given the historical data of yHist.
// yHist: T x K (rows: time, cols: variables). Only last p rows are used as lags.
// steps: number of steps ahead to forecast
// Returns: Steps x K matrix of forecasts
func (rf *ReducedFormVAR) Forecast(yHist *mat.Dense, steps int) (*mat.Dense, error) {
	if rf == nil || len(rf.A) == 0 {
		return nil, fmt.Errorf("VAR model not estimated")
	}
	if steps <= 0 {
		return nil, fmt.Errorf("steps must be > 0")
	}

	p := rf.Model.Lags

	if p <= 0 {
		return nil, fmt.Errorf("lags must be > 0 to forecast")
	}

	// dimensions of yHist, T rows, K cols
	T, K := yHist.Dims()
	if T < p {
		return nil, fmt.Errorf("need at least %d rows in yHist, got %d", p, T)
	}

	totalRows := p + steps

	data := make([]float64, totalRows*K)

	// Gonum.mat stores matrices as a 1d slice at first, so we need to multiply by K to fill out
	for i := 0; i < p; i++ {
		for k := 0; k < K; k++ {
			data[i*K+k] = yHist.At(T-p+i, k)
		}
	}

	out := mat.NewDense(totalRows, K, data)

	// Deterministic structure
	hasConst := rf.Model.Deterministic == DetConst || rf.Model.Deterministic == DetConstTrend
	hasTrend := rf.Model.Deterministic == DetTrend || rf.Model.Deterministic == DetConstTrend

	detConstIdx := 0
	detTrendIdx := 0
	detCols := 0

	if hasConst {
		detCols += 1
	}
	if hasTrend {
		detTrendIdx = detCols
		detCols += 1
	}

	// for each equation in the VAR (for each variable in the time series)
	for step := 0; step < steps; step++ {
		row := p + step
		// time index from last row of yHist
		tIdx := float64(T + step + 1)

		for eq := 0; eq < K; eq++ {
			// val is where we store the equation for the current step for A SINGLE variable
			val := 0.0

			// If model has deterministic trends, include in forecast value by adding it
			if rf.C != nil && detCols > 0 {
				if hasConst {
					val += rf.C.At(eq, detConstIdx)
				}

				if hasTrend {
					val += rf.C.At(eq, detTrendIdx) * tIdx
				}
			}

			// lagged part: sum_j A_j * y_{t-j}
			for lag := 1; lag <= p; lag++ {
				A := rf.A[lag-1]
				prevRow := row - lag
				for j := 0; j < K; j++ {
					val += A.At(eq, j) * out.At(prevRow, j)
				}
			}

			// Sets each row of the forecast with the current value at each column
			out.Set(row, eq, val)
		}
	}
	// Returns only the forecasted rows
	forecast := mat.DenseCopyOf(out.Slice(p, totalRows, 0, K))
	return forecast, nil
}

// IRF computes impulse responses to a one-time structural shock in a variable shockIndex
// Horizon: number of periods to compute (h=0, ..., horizon-1)
// shockIndex: index of variable to shock, (0-based)
// Returns: horizon x K matrix. where row h is response of all K vars at horizon h
func (rf *ReducedFormVAR) IRF(horizon int, shockIndex int) (*mat.Dense, error) {
	if rf == nil || len(rf.A) == 0 {
		return nil, fmt.Errorf("VAR model not estimated")
	}
	if horizon <= 0 {
		return nil, fmt.Errorf("horizon must be > 0")
	}

	p := rf.Model.Lags
	if p <= 0 {
		return nil, fmt.Errorf("lags must be > 0 to IRF")
	}

	K, _ := rf.A[0].Dims()
	if shockIndex < 0 || shockIndex >= K {
		return nil, fmt.Errorf("shockIndex must be between 0 and %d", K-1)
	}

	// Makes the shock matrix
	shock := make([]float64, K)
	if rf.SigmaU != nil {
		var chol mat.Cholesky
		// Cholesky decomposition applied to SigmaU, calculates LL'
		if chol.Factorize(rf.SigmaU) {
			L := mat.NewTriDense(K, mat.Lower, nil)
			chol.LTo(L) // SigmaU = L * L^T
			// get the shock vector
			for i := 0; i < K; i++ {
				shock[i] = L.At(i, shockIndex)
			}
		} else {
			// fallback if SigmaU is not positive definite
			shock[shockIndex] = 1.0
		}
	} else {
		// fall back if SigmaU is not provided
		shock[shockIndex] = 1.0
	}

	// Moving-average coeff matrix Psi_h
	Psi := make([]*mat.Dense, horizon)

	// Psi_0 = I_K, makes matrix using mat
	Idata := make([]float64, K*K)

	for i := 0; i < K; i++ {
		Idata[i*K+i] = 1.0
	}
	// makes a new identity matrix
	Psi[0] = mat.NewDense(K, K, Idata)

	// Recursively computes Psi_h
	for h := 1; h < horizon; h++ {
		M := mat.NewDense(K, K, nil)
		maxLag := p
		if h < p {
			maxLag = h
		}
		for j := 1; j <= maxLag; j++ {
			var tmp mat.Dense
			tmp.Mul(rf.A[j-1], Psi[h-j]) // A_j * Psi_{h-j}
			M.Add(M, &tmp)
		}
		Psi[h] = M
	}

	// IRF[h] = Psi_h * shock

	irf := mat.NewDense(horizon, K, nil)
	shockVec := mat.NewVecDense(K, shock)

	for h := 0; h < horizon; h++ {
		var resp mat.VecDense
		resp.MulVec(Psi[h], shockVec)
		for i := 0; i < K; i++ {
			irf.Set(h, i, resp.AtVec(i))
		}
	}

	return irf, nil
}

// Run IRF for all variables to look for changes in varible var, then compile results
// of how much each varible changed var during its shock into a map
// varIndex: index of variable to analyze, 0-based
// horizon: number of periods to compute (h=0, ..., horizon-1)
// Returns: map[shockIndex] =  impact on varIndex
func (rf *ReducedFormVAR) RunIRFAnalysis(varIndex int, horizon int) (map[int][]float64, error) {
	// Check if the model is estimated and varIndex is valid
	if rf == nil || len(rf.A) == 0 {
		return nil, fmt.Errorf("VAR model not estimated")
	}

	K, _ := rf.A[0].Dims()
	if varIndex < 0 || varIndex >= K {
		return nil, fmt.Errorf("varIndex must be between 0 and %d", K-1)
	}

	results := make(map[int][]float64)
	for shockIdx := 0; shockIdx < K; shockIdx++ {
		irfMat, err := rf.IRF(horizon, shockIdx)
		if err != nil {
			return nil, fmt.Errorf("IRF failed for shockIdx %d: %v", shockIdx, err)
		}

		series := make([]float64, horizon)
		for h := 0; h < horizon; h++ {
			series[h] = irfMat.At(h, varIndex)
		}

		results[shockIdx] = series
	}

	return results, nil
}

// Estimate computes the VAR model parameters using OLS
// ts: TimeSeries struct containing the data
// spec: ModelSpec struct containing the model specification
// opts: EstimationOptions struct containing estimation options
// Returns: ReducedFormVAR struct containing the estimated model
func (e *OLSEstimator) Estimate(ts *TimeSeries, spec ModelSpec, opts EstimationOptions) (*ReducedFormVAR, error) {
	if ts == nil || ts.Y == nil {
		return nil, fmt.Errorf("time series data not provided")
	}

	T, K := ts.Y.Dims()
	p := spec.Lags

	if p <= 0 {
		return nil, fmt.Errorf("lags must be > 0")
	}

	if T <= p {
		return nil, fmt.Errorf("need at least p+1 observations: p = %d, T = %d", p, T)
	}
	if spec.HasExogenous {
		return nil, fmt.Errorf("exogenous variables not supported yet")
	}

	// Builds the response matrix for later use
	Treg := T - p // Usable rows

	// Response matrix Yreg: rows are y_p, y_{p+1}, ..., y_{T-1}
	Yreg := mat.NewDense(Treg, K, nil)
	for t := 0; t < Treg; t++ {
		for k := 0; k < K; k++ {
			Yreg.Set(t, k, ts.Y.At(t+p, k))
		}
	}

	// Deterministic structure
	hasConst := spec.Deterministic == DetConst || spec.Deterministic == DetConstTrend
	hasTrend := spec.Deterministic == DetTrend || spec.Deterministic == DetConstTrend

	detCols := 0
	if hasConst {
		detCols++
	}
	if hasTrend {
		detCols++
	}

	lagCols := p * K
	m := detCols + lagCols // total regressors

	X := mat.NewDense(Treg, m, nil)

	// Fill X row-by-row

	for t := 0; t < Treg; t++ {
		col := 0
		// time index
		timeIndex := float64(t + p + 1)

		if hasConst {
			X.Set(t, col, 1.0)
			col++
		}
		if hasTrend {
			X.Set(t, col, timeIndex)
			col++
		}

		// Lagged Y's: [ y_{t+p-1}, y_{t+p-2}, ..., y_{t+p-p}]
		for j := 1; j <= p; j++ {
			srcRow := t + p - j
			for k := 0; k < K; k++ {
				X.Set(t, col, ts.Y.At(srcRow, k))
				col++
			}
		}
	}

	// B = (X'X)^(-1) X'Y
	// Calculates closed form
	var B mat.Dense

	// First try: normal equations B = (X'X)^(-1) X'Y
	var xtx mat.Dense
	xtx.Mul(X.T(), X)

	var xtxInv mat.Dense

	xtxError := xtxInv.Inverse(&xtx)

	if xtxError == nil {
		// X'X is invertible: standard OLS
		var xty mat.Dense
		xty.Mul(X.T(), Yreg)
		B.Mul(&xtxInv, &xty)
	} else {
		// Fallback: X'X is singular or badly conditioned.
		// Use SVD-based least squares: minimize ||Yreg - X B||_F with minimum-norm B.

		var svd mat.SVD
		ok := svd.Factorize(X, mat.SVDFullU|mat.SVDFullV)
		if !ok {
			return nil, fmt.Errorf("OLS failed: X'X singular and SVD factorization failed: %v", xtxError)
		}

		// Choose an effective numerical rank (tolerance can be tuned)
		rank := svd.Rank(1e-12)

		// Solve X * B ≈ Yreg in least-squares sense; B will be (m × K)
		// This gives us the Moore_penrose pseudoinverse commonly used in regression too
		// If rank == 0, the matrix X is (numerically) all-zero.
		// The minimum-norm least-squares solution to X B ≈ Y is just B = 0.
		if rank == 0 {
			B = *mat.NewDense(m, K, nil) // all zeros
		} else {
			// Solve X * B ≈ Yreg in least-squares sense; B will be (m × K)
			svd.SolveTo(&B, Yreg, rank)
		}
	}

	// Split B into C (deterministic) and A_j's
	var C *mat.Dense
	if detCols > 0 {
		C = mat.NewDense(K, detCols, nil)
		for k := 0; k < K; k++ {
			for d := 0; d < detCols; d++ {
				C.Set(k, d, B.At(d, k))
			}
		}
	}

	A := make([]*mat.Dense, p)
	for j := 0; j < p; j++ {
		Aj := mat.NewDense(K, K, nil)
		rowOffset := detCols + j*K // start row of this lag block in B

		for eq := 0; eq < K; eq++ {
			for colVar := 0; colVar < K; colVar++ {
				Aj.Set(eq, colVar, B.At(rowOffset+colVar, eq))
			}
		}
		A[j] = Aj
	}

	// Residual covariance SigmaU
	var Yhat mat.Dense
	Yhat.Mul(X, &B)

	var U mat.Dense
	U.Sub(Yreg, &Yhat) // Treg x K

	var utu mat.Dense
	utu.Mul(U.T(), &U) // K x K

	df := float64(Treg - m)
	if df <= 0 {
		df = float64(Treg) // fallback
	}

	sigmaData := make([]float64, K*K)
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			sigmaData[i*K+j] = utu.At(i, j) / df
		}
	}
	SigmaU := mat.NewSymDense(K, sigmaData)

	rf := &ReducedFormVAR{
		Model:  spec,
		A:      A,
		C:      C,
		SigmaU: SigmaU,
	}

	return rf, nil
}

// GrangerCausality tests whether causeIdx Granger-causes effectIdx
// in the VAR model. The null hypothesis is that causeIdx does not Granger-cause effectIdx.
// ts: TimeSeries struct containing the data
// causeIdx: index of the variable that may Granger-cause
// effectIdx: index of the variable that may be Granger-caused
// by causeIdx.
// Returns the F-statistic and p-value
func (rf *ReducedFormVAR) GrangerCausality(ts *TimeSeries, causeIdx, effectIdx int) (*GrangerCausalityResult, error) {
	if ts == nil || ts.Y == nil {
		return nil, fmt.Errorf("time series data not provided")
	}
	if rf == nil || len(rf.A) == 0 {
		return nil, fmt.Errorf("VAR model not estimated")
	}

	T, K := ts.Y.Dims()
	p := rf.Model.Lags

	if causeIdx < 0 || causeIdx >= K {
		return nil, fmt.Errorf("causeIdx out of range: %d", causeIdx)
	}
	if effectIdx < 0 || effectIdx >= K {
		return nil, fmt.Errorf("effectIdx out of range: %d", effectIdx)
	}
	if causeIdx == effectIdx {
		return nil, fmt.Errorf("causeIdx and effectIdx cannot be the same")
	}

	// Build response vector for the effect variable: y_effect
	Treg := T - p
	if Treg <= 0 {
		return nil, fmt.Errorf("not enough observations for lags p = %d, T = %d", p, T)
	}
	yEffect := mat.NewVecDense(Treg, nil)
	for t := 0; t < Treg; t++ {
		yEffect.SetVec(t, ts.Y.At(t+p, effectIdx))
	}

	// Deterministic structure (must mirror Estimate)
	hasConst := rf.Model.Deterministic == DetConst || rf.Model.Deterministic == DetConstTrend
	hasTrend := rf.Model.Deterministic == DetTrend || rf.Model.Deterministic == DetConstTrend

	detCols := 0
	if hasConst {
		detCols++
	}
	if hasTrend {
		detCols++
	}

	// --- UNRESTRICTED MODEL (reuse coefficients from rf) ---

	// Build design matrix for unrestricted regression: includes all lagged vars
	lagCols := p * K
	mUnrestricted := detCols + lagCols
	XUnrestricted := mat.NewDense(Treg, mUnrestricted, nil)

	for t := 0; t < Treg; t++ {
		col := 0
		timeIndex := float64(t + p + 1) // same time index convention as Estimate

		if hasConst {
			XUnrestricted.Set(t, col, 1.0)
			col++
		}
		if hasTrend {
			XUnrestricted.Set(t, col, timeIndex)
			col++
		}

		// lag blocks: [ y_{t-1,*}, y_{t-2,*}, ..., y_{t-p,*} ]
		for j := 1; j <= p; j++ {
			srcRow := t + p - j
			for k := 0; k < K; k++ {
				XUnrestricted.Set(t, col, ts.Y.At(srcRow, k))
				col++
			}
		}
	}

	// Construct betaUnrestricted from rf.C and rf.A for the effect equation.
	// Ordering must match XUnrestricted construction above and Estimate()'s B.
	betaUnrestricted := mat.NewVecDense(mUnrestricted, nil)
	coefIndex := 0

	// Deterministic part: C is (K x detCols), row = equation (effectIdx)
	if detCols > 0 && rf.C != nil {
		if hasConst {
			betaUnrestricted.SetVec(coefIndex, rf.C.At(effectIdx, 0))
			coefIndex++
		}
		if hasTrend {
			// If both const & trend, trend is column 1; if only trend, it's column 0.
			trendCol := 0
			if hasConst {
				trendCol = 1
			}
			betaUnrestricted.SetVec(coefIndex, rf.C.At(effectIdx, trendCol))
			coefIndex++
		}
	}

	// Lag blocks: for each lag j, for each variable k, use A_j(effectIdx, k)
	for j := 0; j < p; j++ {
		Aj := rf.A[j] // KxK
		for k := 0; k < K; k++ {
			betaUnrestricted.SetVec(coefIndex, Aj.At(effectIdx, k))
			coefIndex++
		}
	}

	// Sanity check: coefIndex should equal mUnrestricted
	if coefIndex != mUnrestricted {
		return nil, fmt.Errorf("internal error: coefIndex (%d) != mUnrestricted (%d)", coefIndex, mUnrestricted)
	}

	// Compute fitted values and RSS for unrestricted model
	var yHatUnrestricted mat.VecDense
	yHatUnrestricted.MulVec(XUnrestricted, betaUnrestricted)

	var residUnrestricted mat.VecDense
	residUnrestricted.SubVec(yEffect, &yHatUnrestricted)

	rssUnrestricted := mat.Dot(&residUnrestricted, &residUnrestricted)

	// Restricted design matrix: same deterministics, but we skip causeIdx in lag blocks
	mRestricted := detCols + p*(K-1) // exclude p lags of cause variable
	XRestricted := mat.NewDense(Treg, mRestricted, nil)

	for t := 0; t < Treg; t++ {
		col := 0
		timeIndex := float64(t + p + 1)

		if hasConst {
			XRestricted.Set(t, col, 1.0)
			col++
		}
		if hasTrend {
			XRestricted.Set(t, col, timeIndex)
			col++
		}

		for j := 1; j <= p; j++ {
			srcRow := t + p - j
			for k := 0; k < K; k++ {
				if k == causeIdx {
					continue // skip all lags of the cause variable
				}
				XRestricted.Set(t, col, ts.Y.At(srcRow, k))
				col++
			}
		}
	}

	// Fit restricted model via least squares: X_R * beta_R ≈ yEffect
	// We mimic the SVD + pseudoinverse fallback used in Estimate().
	betaRestricted := mat.NewVecDense(mRestricted, nil)

	// First try normal equations: beta = (X'X)^(-1) X'y
	var xtxR mat.Dense
	xtxR.Mul(XRestricted.T(), XRestricted)

	var xtxRInv mat.Dense
	if errInv := xtxRInv.Inverse(&xtxR); errInv == nil {
		// X'X is invertible: standard OLS
		var xtyR mat.Dense

		// yEffect is a vector, turn it into a Treg x 1 matrix for the multiplication
		yMat := mat.NewDense(Treg, 1, nil)
		for t := 0; t < Treg; t++ {
			yMat.Set(t, 0, yEffect.AtVec(t))
		}

		xtyR.Mul(XRestricted.T(), yMat) // (mRestricted x Treg) * (Treg x 1) = (mRestricted x 1)

		var b mat.Dense
		b.Mul(&xtxRInv, &xtyR) // (mRestricted x mRestricted) * (mRestricted x 1)

		for i := 0; i < mRestricted; i++ {
			betaRestricted.SetVec(i, b.At(i, 0))
		}
	} else {
		// Fallback: X'X is singular or badly conditioned. Use SVD-based least squares.
		var svd mat.SVD
		ok := svd.Factorize(XRestricted, mat.SVDFullU|mat.SVDFullV)
		if !ok {
			return nil, fmt.Errorf("restricted OLS failed: X'X singular and SVD factorization failed: %v", errInv)
		}

		rank := svd.Rank(1e-12)

		if rank == 0 {
			// Everything is (numerically) zero – minimum-norm solution is beta = 0,
			// which we already have in betaRestricted (all zeros).
		} else {
			// Solve X_R * beta ≈ yEffect in least-squares sense.
			yMat := mat.NewDense(Treg, 1, nil)
			for t := 0; t < Treg; t++ {
				yMat.Set(t, 0, yEffect.AtVec(t))
			}

			var b mat.Dense // (mRestricted x 1)
			svd.SolveTo(&b, yMat, rank)

			for i := 0; i < mRestricted; i++ {
				betaRestricted.SetVec(i, b.At(i, 0))
			}
		}
	}

	// Compute fitted values and RSS for restricted model
	var yHatRestricted mat.VecDense
	yHatRestricted.MulVec(XRestricted, betaRestricted)

	var residRestricted mat.VecDense
	residRestricted.SubVec(yEffect, &yHatRestricted)

	rssRestricted := mat.Dot(&residRestricted, &residRestricted)

	// F-statistic and p-value
	q := float64(p)             // number of restrictions
	k := float64(mUnrestricted) // parameters in unrestricted model
	dof := float64(Treg) - k    // denominator degrees of freedom

	if dof <= 0 {
		return nil, fmt.Errorf("insufficient degrees of freedom: %f", dof)
	}

	// Protect against numerical issues:
	// In theory rssRestricted >= rssUnrestricted, but due to floating point
	// we can get a tiny negative difference.
	num := rssRestricted - rssUnrestricted
	if num < 0 {
		num = 0 // clamp to zero
	}

	den := rssUnrestricted / dof
	var fStatistic float64
	var pValue float64

	if den <= 0 || num == 0 {
		// If the unrestricted RSS is zero or the difference is zero/negative,
		// there's no evidence that the extra lags matter.
		fStatistic = 0
		pValue = 1
	} else {
		fStatistic = (num / q) / den

		// Guard before calling F.CDF: F is only defined for x >= 0.
		if fStatistic <= 0 || math.IsNaN(fStatistic) || math.IsInf(fStatistic, 0) {
			fStatistic = 0
			pValue = 1
		} else {
			fDist := distuv.F{
				D1: q,
				D2: dof,
			}
			pValue = 1.0 - fDist.CDF(fStatistic)
		}
	}

	// Final sanity clamp on pValue to ensure it's in [0, 1]
	if pValue < 0 {
		pValue = 0
	}
	if pValue > 1 {
		pValue = 1
	}

	result := &GrangerCausalityResult{
		CauseVar:    ts.VarNames[causeIdx],
		EffectVar:   ts.VarNames[effectIdx],
		FStatistic:  fStatistic,
		PValue:      pValue,
		Lags:        p,
		Significant: pValue < 0.05,
	}

	return result, nil
}

// GrangerCausalityMatrix performs pairwise Granger causality tests for all variables
// in the VAR model and returns a matrix of GrangerCausalityResult.
// Returns: K x K matrix of GrangerCausalityResult, where result[i][j] is the result
func (rf *ReducedFormVAR) GrangerCausalityMatrix(ts *TimeSeries) ([][]*GrangerCausalityResult, error) {
	if ts == nil || ts.Y == nil {
		return nil, fmt.Errorf("time series data not provided")
	}

	_, K := ts.Y.Dims()

	// Create matrix to store results
	results := make([][]*GrangerCausalityResult, K)
	for i := range results {
		results[i] = make([]*GrangerCausalityResult, K)
	}

	// Perform pairwise tests
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			if i == j {
				// No self-causality test
				results[i][j] = nil
				continue
			}

			result, err := rf.GrangerCausality(ts, i, j)
			if err != nil {
				return nil, fmt.Errorf("error testing %s -> %s: %v", ts.VarNames[i], ts.VarNames[j], err)
			}
			results[i][j] = result
		}
	}

	return results, nil
}

// This function takes in the created Granger Matrix and outputs it to a CSV file with
// the columns: CauseVar, EffectVar, FStatistic, PValue, Lags, Significant
// Returns an error if the file cannot be written, otherwise returns nil and writes the file
func (rf *ReducedFormVAR) OutputGrangerMatrixToCSV(path string, gcMatrix [][]*GrangerCausalityResult, varNames []string) error {
	file, err := os.Create(path)

	if err != nil {
		return err
	}

	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := []string{"CauseVar", "EffectVar", "FStatistic", "PValue", "Lags", "Significant"}
	if err := writer.Write(header); err != nil {
		return err
	}

	K := len(varNames)

	// Write data rows
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			if i == j {
				continue // Skip self-causality
			}
			result := gcMatrix[i][j]
			if result == nil {
				continue
			}
			record := []string{
				result.CauseVar,
				result.EffectVar,
				fmt.Sprintf("%f", result.FStatistic),
				fmt.Sprintf("%f", result.PValue),
				fmt.Sprintf("%d", result.Lags),
				fmt.Sprintf("%t", result.Significant),
			}
			if err := writer.Write(record); err != nil {
				return err
			}
		}
	}

	return nil
}

// --- Bootstrapping below ---

// computeResiduals recomputes residuals U (T-p x K) from a fitted VAR and data ts.
// It uses the same deterministic structure as Estimate().
func (rf *ReducedFormVAR) computeResiduals(ts *TimeSeries) (*mat.Dense, error) {
	if ts == nil || ts.Y == nil {
		return nil, fmt.Errorf("time series data not provided")
	}
	if rf == nil || len(rf.A) == 0 {
		return nil, fmt.Errorf("VAR model not estimated")
	}

	T, K := ts.Y.Dims()
	p := rf.Model.Lags
	if T <= p {
		return nil, fmt.Errorf("need at least p+1 observations: p = %d, T = %d", p, T)
	}

	hasConst := rf.Model.Deterministic == DetConst || rf.Model.Deterministic == DetConstTrend
	hasTrend := rf.Model.Deterministic == DetTrend || rf.Model.Deterministic == DetConstTrend

	detCols := 0
	if hasConst {
		detCols++
	}
	if hasTrend {
		detCols++
	}

	// residuals for rows t = p,...,T-1  => T-p rows
	Treg := T - p
	U := mat.NewDense(Treg, K, nil)

	for t := p; t < T; t++ {
		timeIndex := float64(t + 1) // matches Estimate's time index convention

		for eq := 0; eq < K; eq++ {
			val := 0.0

			// deterministic terms
			if detCols > 0 && rf.C != nil {
				detIdx := 0
				if hasConst {
					val += rf.C.At(eq, detIdx)
					detIdx++
				}
				if hasTrend {
					val += rf.C.At(eq, detIdx) * timeIndex
				}
			}

			// lag terms: sum_{j=1}^p A_j(eq, :) y_{t-j}
			for j := 1; j <= p; j++ {
				Aj := rf.A[j-1]
				srcRow := t - j
				for k := 0; k < K; k++ {
					val += Aj.At(eq, k) * ts.Y.At(srcRow, k)
				}
			}

			// residual = observed - fitted
			u := ts.Y.At(t, eq) - val
			U.Set(t-p, eq, u)
		}
	}

	return U, nil
}

// simulateBootstrapSeries generates a bootstrap sample Y* of the same length as ts,
// using the fitted VAR coefficients and residuals resU (T-p x K), where each row
// is a residual vector. We resample rows of resU with replacement.
func (rf *ReducedFormVAR) simulateBootstrapSeries(
	ts *TimeSeries,
	resU *mat.Dense,
	rng *rand.Rand,
) (*TimeSeries, error) {

	if ts == nil || ts.Y == nil {
		return nil, fmt.Errorf("time series data not provided")
	}
	if rf == nil || len(rf.A) == 0 {
		return nil, fmt.Errorf("VAR model not estimated")
	}

	T, K := ts.Y.Dims()
	p := rf.Model.Lags
	if T <= p {
		return nil, fmt.Errorf("need at least p+1 observations: p = %d, T = %d", p, T)
	}

	Treg, kRes := resU.Dims()
	if Treg != T-p || kRes != K {
		return nil, fmt.Errorf("residual matrix has wrong shape: got %dx%d, expected %dx%d",
			Treg, kRes, T-p, K)
	}

	hasConst := rf.Model.Deterministic == DetConst || rf.Model.Deterministic == DetConstTrend
	hasTrend := rf.Model.Deterministic == DetTrend || rf.Model.Deterministic == DetConstTrend

	detCols := 0
	if hasConst {
		detCols++
	}
	if hasTrend {
		detCols++
	}

	// Prepare bootstrap residuals ε*_t for t = p,...,T-1 (T-p rows)
	epsBoot := mat.NewDense(Treg, K, nil)
	for i := 0; i < Treg; i++ {
		// resample residual row index from [0, Treg-1]
		idx := rng.Intn(Treg)
		for j := 0; j < K; j++ {
			epsBoot.Set(i, j, resU.At(idx, j))
		}
	}

	// Simulate Y*, same dimension as original
	Ystar := mat.NewDense(T, K, nil)

	// Copy the first p observations from original data
	for t := 0; t < p; t++ {
		for j := 0; j < K; j++ {
			Ystar.Set(t, j, ts.Y.At(t, j))
		}
	}

	// Simulate t = p,...,T-1
	for t := p; t < T; t++ {
		timeIndex := float64(t + 1)
		epsRow := t - p // index into epsBoot

		for eq := 0; eq < K; eq++ {
			val := 0.0

			// deterministic terms
			if detCols > 0 && rf.C != nil {
				detIdx := 0
				if hasConst {
					val += rf.C.At(eq, detIdx)
					detIdx++
				}
				if hasTrend {
					val += rf.C.At(eq, detIdx) * timeIndex
				}
			}

			// lag terms: sum_{j=1}^p A_j(eq,:) y*_{t-j}
			for j := 1; j <= p; j++ {
				Aj := rf.A[j-1]
				srcRow := t - j
				for k := 0; k < K; k++ {
					val += Aj.At(eq, k) * Ystar.At(srcRow, k)
				}
			}

			// add bootstrap residual
			val += epsBoot.At(epsRow, eq)
			Ystar.Set(t, eq, val)
		}
	}

	// Preserve time index and variable names
	times := make([]float64, T)
	if len(ts.Time) == T {
		copy(times, ts.Time)
	} else {
		// fallback to simple 0,1,2,... if original Time is missing
		for i := 0; i < T; i++ {
			times[i] = float64(i)
		}
	}

	tsStar := &TimeSeries{
		Y:        Ystar,
		Time:     times,
		VarNames: ts.VarNames,
	}
	return tsStar, nil
}

// bootstrapQuantile returns the empirical q-quantile of samples (0 <= q <= 1)
// using linear interpolation between order statistics.
func bootstrapQuantile(samples []float64, q float64) float64 {
	n := len(samples)
	if n == 0 {
		return math.NaN()
	}

	tmp := make([]float64, n)
	copy(tmp, samples)
	sort.Float64s(tmp)

	if q <= 0 {
		return tmp[0]
	}
	if q >= 1 {
		return tmp[n-1]
	}

	pos := q * float64(n-1)
	idxBelow := int(math.Floor(pos))
	idxAbove := int(math.Ceil(pos))

	if idxAbove == idxBelow {
		return tmp[idxBelow]
	}

	weight := pos - float64(idxBelow)
	return tmp[idxBelow]*(1.0-weight) + tmp[idxAbove]*weight
}

// BootstrapIRF performs a residual bootstrap for IRFs for all shock variables.
// Returns a map[shockIndex]*IRFBootstrapResult, where each result contains
// the point estimate IRF and lower/upper CI bands.
func (rf *ReducedFormVAR) BootstrapIRF(
	ts *TimeSeries,
	opts BootstrapOptions,
) (map[int]*IRFBootstrapResult, error) {

	if ts == nil || ts.Y == nil {
		return nil, fmt.Errorf("time series data not provided")
	}
	if rf == nil || len(rf.A) == 0 {
		return nil, fmt.Errorf("VAR model not estimated")
	}

	// Default options if not set
	if opts.NReplications <= 0 {
		opts.NReplications = 500
	}
	if opts.Horizon <= 0 {
		opts.Horizon = 12
	}
	if opts.Alpha <= 0 || opts.Alpha >= 1 {
		opts.Alpha = 0.05
	}

	T, K := ts.Y.Dims()
	if T <= rf.Model.Lags {
		return nil, fmt.Errorf("not enough data: T=%d, p=%d", T, rf.Model.Lags)
	}

	H := opts.Horizon

	// 1. Compute original (point estimate) IRFs for each shock variable
	results := make(map[int]*IRFBootstrapResult, K)
	// For collecting bootstrap samples: shockIdx -> [h][var][]values
	shockIRFValues := make(map[int][][][]float64, K)

	for shockIdx := 0; shockIdx < K; shockIdx++ {
		baseIRF, err := rf.IRF(H, shockIdx)
		if err != nil {
			return nil, fmt.Errorf("IRF failed for shock %d on original model: %v", shockIdx, err)
		}

		res := &IRFBootstrapResult{
			ShockIndex: shockIdx,
			Horizon:    H,
			Alpha:      opts.Alpha,
			Point:      baseIRF,
			Lower:      mat.NewDense(H, K, nil),
			Upper:      mat.NewDense(H, K, nil),
		}
		results[shockIdx] = res

		// Set up storage for bootstrap draws
		vals := make([][][]float64, H)
		for h := 0; h < H; h++ {
			vals[h] = make([][]float64, K)
			for j := 0; j < K; j++ {
				vals[h][j] = make([]float64, 0, opts.NReplications)
			}
		}
		shockIRFValues[shockIdx] = vals
	}

	// 2. Compute residuals from original model
	resU, err := rf.computeResiduals(ts)
	if err != nil {
		return nil, fmt.Errorf("failed to compute residuals: %v", err)
	}

	// 3. Prepare per-replication seeds (so RNG is not shared across goroutines)
	var masterSeed int64
	if opts.Seed != 0 {
		masterSeed = opts.Seed
	} else {
		masterSeed = time.Now().UnixNano()
	}
	masterRng := rand.New(rand.NewSource(masterSeed))

	seeds := make([]int64, opts.NReplications)
	for i := 0; i < opts.NReplications; i++ {
		seeds[i] = masterRng.Int63()
	}

	// 4. Set up worker pool for parallel bootstrapping
	numWorkers := runtime.NumCPU()
	if numWorkers > opts.NReplications {
		numWorkers = opts.NReplications
	}

	jobs := make(chan int)
	resultsCh := make(chan irfReplication, opts.NReplications)

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	// Worker function
	worker := func() {
		defer wg.Done()

		for b := range jobs {
			// Local RNG for this replication
			rng := rand.New(rand.NewSource(seeds[b]))

			// 4a. Simulate a bootstrap sample Y*
			tsStar, errSim := rf.simulateBootstrapSeries(ts, resU, rng)
			if errSim != nil {
				// In a production library you'd want better error handling,
				// but panic is OK for now to reveal issues.
				panic(fmt.Errorf("bootstrap %d: simulate failed: %v", b, errSim))
			}

			// 4b. Re-estimate VAR on Y*
			bootRF, errEst := (&OLSEstimator{}).Estimate(tsStar, rf.Model, EstimationOptions{})
			if errEst != nil {
				panic(fmt.Errorf("bootstrap %d: VAR estimation failed: %v", b, errEst))
			}

			// 4c. Compute IRFs for each shock variable under bootstrap model
			rep := irfReplication{
				ShockIRFs: make(map[int][][]float64, K),
			}

			for shockIdx := 0; shockIdx < K; shockIdx++ {
				irfBoot, errIRF := bootRF.IRF(H, shockIdx)
				if errIRF != nil {
					panic(fmt.Errorf("bootstrap %d: IRF failed for shock %d: %v", b, shockIdx, errIRF))
				}

				hRows, kCols := irfBoot.Dims()
				m := make([][]float64, hRows)
				for h := 0; h < hRows; h++ {
					m[h] = make([]float64, kCols)
					for j := 0; j < kCols; j++ {
						m[h][j] = irfBoot.At(h, j)
					}
				}
				rep.ShockIRFs[shockIdx] = m
			}

			// Send this replication's IRFs to the aggregator
			resultsCh <- rep
		}
	}

	// Start workers
	for w := 0; w < numWorkers; w++ {
		go worker()
	}

	// Feed jobs
	go func() {
		for b := 0; b < opts.NReplications; b++ {
			jobs <- b
		}
		close(jobs)
	}()

	// 5. Aggregator: collect all replication results and fill shockIRFValues
	for i := 0; i < opts.NReplications; i++ {
		rep := <-resultsCh

		for shockIdx, matIRF := range rep.ShockIRFs {
			vals := shockIRFValues[shockIdx]
			for h := 0; h < H; h++ {
				for j := 0; j < K; j++ {
					vals[h][j] = append(vals[h][j], matIRF[h][j])
				}
			}
		}
	}

	// All results collected; workers can be joined now
	wg.Wait()
	close(resultsCh)

	// 6. Compute CI bands from bootstrap distributions
	lowerQ := opts.Alpha / 2.0
	upperQ := 1.0 - opts.Alpha/2.0

	for shockIdx, res := range results {
		vals := shockIRFValues[shockIdx]
		for h := 0; h < H; h++ {
			for j := 0; j < K; j++ {
				samples := vals[h][j]
				if len(samples) == 0 {
					res.Lower.Set(h, j, math.NaN())
					res.Upper.Set(h, j, math.NaN())
					continue
				}
				lo := bootstrapQuantile(samples, lowerQ)
				hi := bootstrapQuantile(samples, upperQ)
				res.Lower.Set(h, j, lo)
				res.Upper.Set(h, j, hi)
			}
		}
	}

	return results, nil
}

// irfReplication holds the IRF matrices for all shocks from a single bootstrap replication.
// ShockIRFs[shockIdx][h][j] = IRF(h, j) for that replication.
type irfReplication struct {
	ShockIRFs map[int][][]float64
}

// BootstrapGrangerMatrix performs a residual bootstrap for Granger causality
// for all variable pairs (i -> j, i != j).
// It returns a K x K matrix of GrangerCausalityBootstrapResult, where
// result[i][j] is nil if i == j.
func (rf *ReducedFormVAR) BootstrapGrangerMatrix(
	ts *TimeSeries,
	opts GrangerBootstrapOptions,
) ([][]*GrangerCausalityBootstrapResult, error) {

	if ts == nil || ts.Y == nil {
		return nil, fmt.Errorf("time series data not provided")
	}
	if rf == nil || len(rf.A) == 0 {
		return nil, fmt.Errorf("VAR model not estimated")
	}

	// Defaults
	if opts.NReplications <= 0 {
		opts.NReplications = 500
	}
	if opts.Alpha <= 0 || opts.Alpha >= 1 {
		opts.Alpha = 0.05
	}

	T, K := ts.Y.Dims()
	if T <= rf.Model.Lags {
		return nil, fmt.Errorf("not enough data: T=%d, p=%d", T, rf.Model.Lags)
	}

	// 1. Original analytic Granger matrix
	baseGC, err := rf.GrangerCausalityMatrix(ts)
	if err != nil {
		return nil, fmt.Errorf("failed to compute base Granger matrix: %v", err)
	}

	// 2. Residuals from original VAR
	resU, err := rf.computeResiduals(ts)
	if err != nil {
		return nil, fmt.Errorf("failed to compute residuals: %v", err)
	}

	// 3. RNG seeding
	var seed int64
	if opts.Seed != 0 {
		seed = opts.Seed
	} else {
		seed = time.Now().UnixNano()
	}
	masterRng := rand.New(rand.NewSource(seed))

	// Per-replication seeds so workers don't share RNG
	seeds := make([]int64, opts.NReplications)
	for i := 0; i < opts.NReplications; i++ {
		seeds[i] = masterRng.Int63()
	}

	// 4. Counts for bootstrap p-values
	counts := make([][]int, K)
	for i := 0; i < K; i++ {
		counts[i] = make([]int, K)
	}

	// 5. Worker pool setup
	numWorkers := runtime.NumCPU()
	if numWorkers > opts.NReplications {
		numWorkers = opts.NReplications
	}

	jobs := make(chan int)
	resultsCh := make(chan gcReplication, opts.NReplications)

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	worker := func() {
		defer wg.Done()
		for b := range jobs {
			rng := rand.New(rand.NewSource(seeds[b]))

			// 5a. Simulate bootstrap sample Y*
			tsStar, errSim := rf.simulateBootstrapSeries(ts, resU, rng)
			if errSim != nil {
				panic(fmt.Errorf("bootstrap %d: simulate failed: %v", b, errSim))
			}

			// 5b. Re-estimate VAR on Y*
			bootRF, errEst := (&OLSEstimator{}).Estimate(tsStar, rf.Model, EstimationOptions{})
			if errEst != nil {
				panic(fmt.Errorf("bootstrap %d: VAR estimation failed: %v", b, errEst))
			}

			// 5c. Compute Granger matrix on bootstrap sample
			bootGC, errGC := bootRF.GrangerCausalityMatrix(tsStar)
			if errGC != nil {
				panic(fmt.Errorf("bootstrap %d: Granger matrix failed: %v", b, errGC))
			}

			// 5d. Extract F-stats into a dense KxK matrix
			F := make([][]float64, K)
			for i := 0; i < K; i++ {
				F[i] = make([]float64, K)
				for j := 0; j < K; j++ {
					if i == j {
						F[i][j] = 0.0
						continue
					}
					if bootGC[i][j] != nil {
						F[i][j] = bootGC[i][j].FStatistic
					} else {
						F[i][j] = 0.0
					}
				}
			}

			resultsCh <- gcReplication{FStats: F}
		}
	}

	// Start workers
	for w := 0; w < numWorkers; w++ {
		go worker()
	}

	// Feed jobs
	go func() {
		for b := 0; b < opts.NReplications; b++ {
			jobs <- b
		}
		close(jobs)
	}()

	// 6. Aggregator: collect replication results and update counts
	for i := 0; i < opts.NReplications; i++ {
		rep := <-resultsCh
		Fboot := rep.FStats

		for c := 0; c < K; c++ {
			for e := 0; e < K; e++ {
				if c == e {
					continue
				}
				base := baseGC[c][e]
				if base == nil {
					continue
				}
				if Fboot[c][e] >= base.FStatistic {
					counts[c][e]++
				}
			}
		}
	}

	wg.Wait()
	close(resultsCh)

	// 7. Build output matrix with bootstrap p-values
	out := make([][]*GrangerCausalityBootstrapResult, K)
	for i := 0; i < K; i++ {
		out[i] = make([]*GrangerCausalityBootstrapResult, K)
		for j := 0; j < K; j++ {
			if i == j {
				out[i][j] = nil
				continue
			}

			base := baseGC[i][j]
			if base == nil {
				out[i][j] = nil
				continue
			}

			// small-sample correction: (count+1)/(N+1)
			bootP := float64(counts[i][j]+1) / float64(opts.NReplications+1)

			res := &GrangerCausalityBootstrapResult{
				Base:        base,
				BootPValue:  bootP,
				Alpha:       opts.Alpha,
				Significant: bootP < opts.Alpha,
			}
			out[i][j] = res
		}
	}

	return out, nil
}
