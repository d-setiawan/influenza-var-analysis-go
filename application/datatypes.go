// Authors: Rohan Adla, Arrio Gonsalves, Shreyan Nalwad, Dylan Setiawan
// Date: Dec 12th 2025
// Project: A VAR-based Computational Analysis of Influenza and Weather Dynamics
// Class: 02-613 at Caregie Mellon University

package main

import (
	"gonum.org/v1/gonum/mat"
)

// Simple struct for time series data
type TimeSeries struct {
	// Matrix for data
	Y *mat.Dense
	// Tracks number of time points, basically rows
	Time []float64
	// List of variable Names
	VarNames []string
}

// What kind of constant to include in the model
type Deterministic int

// Deterministic Constants for VAR
const (
	DetNone Deterministic = iota
	DetConst
	DetTrend
	DetConstTrend
)

// What kind of model to fit
type ModelSpec struct {
	// How many lags?
	Lags int
	// What kind of constant to include
	Deterministic Deterministic
	// Does it have extra variables?
	HasExogenous bool
}

// ReducedFormVAR represents the reduced form of a VAR model.
type ReducedFormVAR struct {
	Model ModelSpec

	// Coefficient matrices for each lag A_1, A_2, etc (each KxK matrix)
	// Stored as a slice of matrices
	A []*mat.Dense

	// Deterministic Terms: e.g. constant (Kx1) and trend (Kx1) if included
	C *mat.Dense

	// Covariance of residuals (KxK)
	SigmaU *mat.SymDense
}

// ReducedForm is the interface for a reduced form VAR model.
type ReducedForm interface {
	// Returns the model specification
	Spec() ModelSpec
	// Returns the coefficient matrices
	Phi() []*mat.Dense
	// Returns the error covariance
	CovU() *mat.SymDense

	// compute the forcasts for a given initial state
	Forecast(y0 *mat.Dense, steps int) (*mat.Dense, error)
	// Simulates effect of one-time shock in 1 variable on all variables over time
	IRF(horizon int, shockIndex int) (*mat.Dense, error)
	// New: residual bootstrap for IRFs
	BootstrapIRF(ts *TimeSeries, opts BootstrapOptions) (map[int]*IRFBootstrapResult, error)
}

// EstimationOptions contains options like regularization strngth, priors, etc.
// Added for future expansion and modularity, no use right now
type EstimationOptions struct {
	// For standard VAr
	UseGeneralizedLeastSquares bool
}

// Estimator is the interface for a VAR model estimator.
type Estimator interface {
	// Turns the data we have into a reduced form VAR
	Estimate(ts *TimeSeries, spec ModelSpec, opts EstimationOptions) (*ReducedFormVAR, error)
}

// OLSEstimator implements the OLS estimator for VAR models.
type OLSEstimator struct{}

// VARModel holds the results of fitting a VAR model.
type VARModel struct {
	LagP         int         // Optimal lag order used (p)
	Coefficients [][]float64 // The fitted A_1...A_p matrices (the core model parameters)
	Residuals    []float64   // Model residuals (for checking assumptions)
	Variables    []string    // List of variables included in the model (e.g., ["A_H1N1_Count", "Avg_Temperature"])

	// Granger Causality Results
	GrangerPValues map[string]map[string]float64 // Map[CauseVar][EffectVar] = PValue
}

// --- GRANGER CAUSALITY TEST ---

// GrangerCausalityResult holds the result of a Granger causality test
type GrangerCausalityResult struct {
	CauseVar    string  // Variable being tested as the cause
	EffectVar   string  // Variable being tested as the effect
	FStatistic  float64 // F-statistic value
	PValue      float64 // P-value
	Lags        int     // Number of lags used
	Significant bool    // True if p-value < 0.05
}

// Options for bootstrap IRFs
type BootstrapOptions struct {
	// Number of bootstrap replications (e.g., 500–2000)
	NReplications int

	// Horizon for IRFs (number of periods h = 0,...,H-1)
	Horizon int

	// Confidence level alpha (e.g., 0.05 for 95% CI)
	Alpha float64

	// RNG seed (if 0, time-based seed is used)
	Seed int64
}

// IRFBootstrapResult stores point estimates and CI bands for one shock.
type IRFBootstrapResult struct {
	ShockIndex int     // which variable was shocked
	Horizon    int     // number of IRF periods
	Alpha      float64 // significance level (e.g. 0.05)

	// Point estimate IRF (horizon x K), from original rf.IRF(...)
	Point *mat.Dense

	// Lower and Upper CI bands (same dimensions as Point)
	Lower *mat.Dense
	Upper *mat.Dense
}

// Options for bootstrap Granger causality specifically.
type GrangerBootstrapOptions struct {
	NReplications int     // e.g. 500–2000
	Alpha         float64 // e.g. 0.05 for 95% significance
	Seed          int64   // RNG seed; 0 = time-based
}

// GrangerCausalityBootstrapResult holds the results of a bootstrap Granger causality test.
type GrangerCausalityBootstrapResult struct {
	Base        *GrangerCausalityResult // original analytic GC result
	BootPValue  float64                 // bootstrap p-value
	Alpha       float64                 // significance level used
	Significant bool                    // BootPValue < Alpha
}

// gcReplication holds the F-statistics from one bootstrap replication.
type gcReplication struct {
	FStats [][]float64
}
