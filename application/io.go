// Authors: Rohan Adla, Arrio Gonsalves, Shreyan Nalwad, Dylan Setiawan
// Date: Dec 12th 2025
// Project: A VAR-based Computational Analysis of Influenza and Weather Dynamics
// Class: 02-613 at Caregie Mellon University

package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// LoadCSVToTimeSeries loads a CSV file into a TimeSeries struct.
func LoadCSVToTimeSeries(path string) (*TimeSeries, error) {
	// 1. Open file
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	// 2. Make CSV reader
	r := csv.NewReader(f)
	r.TrimLeadingSpace = true

	// 3. Read header row
	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if len(header) == 0 {
		return nil, fmt.Errorf("empty header in %s", path)
	}
	K := len(header) // number of variables

	var (
		data  []float64 // flat data for mat.Dense
		times []float64 // time index
		row   int       // row counter
	)

	// 4. Read each data row
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read row %d: %w", row+2, err) // +2 for header + 1-based
		}

		// Skip completely empty lines (optional, but nice to have)
		if len(record) == 1 && record[0] == "" {
			continue
		}

		if len(record) != K {
			return nil, fmt.Errorf(
				"row %d: expected %d columns, got %d",
				row+2, K, len(record),
			)
		}

		for j, s := range record {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return nil, fmt.Errorf(
					"parse float at row %d col %d (%q): %w",
					row+2, j+1, s, err,
				)
			}
			data = append(data, v)
		}

		// Here we just use a simple time index: 0,1,2,...
		times = append(times, float64(row))
		row++
	}

	if row == 0 {
		return nil, fmt.Errorf("no data rows in %s", path)
	}

	T := row

	// 5. Build mat.Dense
	Y := mat.NewDense(T, K, data)

	// 6. Build TimeSeries
	ts := &TimeSeries{
		Y:        Y,
		Time:     times,
		VarNames: header,
	}

	return ts, nil
}

// Helper function to print coefficient matrices
func (rf *ReducedFormVAR) PrintCoefficients() {
	for i, Ai := range rf.A {
		fmt.Printf("\n=== A_%d ===\n", i+1)
		fmt.Printf("%v\n", mat.Formatted(Ai, mat.Prefix(" ")))
	}

	fmt.Println("\n=== Covariance Matrix Σ_u ===")
	fmt.Printf("%v\n", mat.Formatted(rf.SigmaU, mat.Prefix(" ")))
}

// Helper function to print forecasts
func PrintForecast(fc *mat.Dense) {
	fmt.Println("\n=== Forecast Matrix ===")
	fmt.Printf("%v\n", mat.Formatted(fc, mat.Prefix(" ")))
}

// Helps print the IRF matrix, requires the matrix, variable names, and the shockindex
func PrintIRF(irf *mat.Dense, varNames []string, shockIndex int) {
	rows, cols := irf.Dims()

	fmt.Printf("\n=== Impulse Response Function ===\n")
	fmt.Printf("Shock to variable %d (%s)\n\n", shockIndex, varNames[shockIndex])

	// Print header
	fmt.Printf("h\t")
	for _, name := range varNames {
		fmt.Printf("%12s", name)
	}
	fmt.Println()

	// Print rows
	for h := 0; h < rows; h++ {
		fmt.Printf("%d\t", h)
		for j := 0; j < cols; j++ {
			fmt.Printf("%12.6f", irf.At(h, j))
		}
		fmt.Println()
	}
}

// Produces a summary table of all of the model params
func (rf *ReducedFormVAR) Summary(ts *TimeSeries) {
	// Check if rf is nil
	if rf == nil {
		fmt.Println("VAR model is nil")
		return
	}
	fmt.Println("         Reduced-form VAR Summary      ")

	// Basic dimensions
	var (
		T, K int
	)

	if ts != nil && ts.Y != nil {
		T, K = ts.Y.Dims()
	} else if len(rf.A) > 0 && rf.A[0] != nil {
		K, _ = rf.A[0].Dims()
	}

	p := rf.Model.Lags

	fmt.Printf("Number of variables (K): %d\n", K)
	fmt.Printf("Lag order (p):           %d\n", p)
	if ts != nil && ts.Y != nil {
		fmt.Printf("Sample size (T):         %d\n", T)
	}
	fmt.Printf("Number of lag matrices:  %d\n", len(rf.A))
	fmt.Println()

	// Model specifications
	fmt.Println("Model specification:")
	fmt.Printf("  Deterministic: %v\n", rf.Model.Deterministic)
	fmt.Printf("  Has exogenous: %v\n", rf.Model.HasExogenous)
	fmt.Println()

	if ts != nil && len(ts.VarNames) > 0 {
		fmt.Println("Variables:")
		fmt.Printf("  %s\n", strings.Join(ts.VarNames, ", "))
		fmt.Println()
	}

	// Parameter counts
	numCoeff := 0
	if len(rf.A) > 0 && rf.A[0] != nil {
		rA, cA := rf.A[0].Dims()
		numCoeff = len(rf.A) * rA * cA
	}
	fmt.Printf("Total coefficients in A matrices: %d\n", numCoeff)

	// intercepts (if any)
	if rf.C != nil {
		rC, cC := rf.C.Dims()
		fmt.Printf("Intercept terms (C) dimensions:   %d x %d\n", rC, cC)
	}
	fmt.Println()

	// Coefficient matrices
	if len(rf.A) > 0 {
		fmt.Println("Coefficient matrices A_1 ... A_p:")
		for i, Ai := range rf.A {
			if Ai == nil {
				continue
			}
			fmt.Printf("\nA_%d =\n", i+1)
			fmt.Printf("%v\n", mat.Formatted(Ai, mat.Prefix("  ")))
		}
		fmt.Println()
	}

	// Intercept (if present)
	if rf.C != nil {
		fmt.Println("Intercept matrix C:")
		fmt.Printf("%v\n", mat.Formatted(rf.C, mat.Prefix("  ")))
		fmt.Println()
	}

	// Covariance matrix Σ_u
	if rf.SigmaU != nil {
		fmt.Println("Residual covariance matrix Σ_u:")
		fmt.Printf("%v\n", mat.Formatted(rf.SigmaU, mat.Prefix("  ")))
		fmt.Println()
	}

	fmt.Println("=======================================")
}

// PrintGrangerCausality prints the Granger causality test results in a formatted table
func PrintGrangerCausality(results [][]*GrangerCausalityResult, varNames []string) {
	fmt.Println("\n=== Granger Causality Test Results ===")
	fmt.Println("Null Hypothesis: Variable X does NOT Granger-cause Variable Y")
	fmt.Println("Significance level: α = 0.05")
	fmt.Println()

	K := len(varNames)

	// Print header
	fmt.Printf("%-20s -> %-20s | F-Statistic | P-Value  | Conclusion\n", "Cause", "Effect")
	fmt.Println("------------------------------------------------------------------------------------")

	// Print results
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			if i == j {
				continue
			}

			result := results[i][j]
			if result == nil {
				continue
			}

			conclusion := "No causality"
			if result.Significant {
				conclusion = "GRANGER-CAUSES"
			}

			fmt.Printf("%-20s -> %-20s | %11.4f | %8.6f | %s\n",
				result.CauseVar,
				result.EffectVar,
				result.FStatistic,
				result.PValue,
				conclusion)
		}
	}
	fmt.Println()
}

// OutputBootstrapIRFToCSV writes bootstrap IRF results to CSV in long format.
// Columns: ShockVar, ResponseVar, Horizon, Point, Lower, Upper
func OutputBootstrapIRFToCSV(
	path string,
	boot map[int]*IRFBootstrapResult,
	varNames []string,
) error {

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{"ShockVar", "ResponseVar", "Horizon", "Point", "Lower", "Upper"}
	if err := writer.Write(header); err != nil {
		return err
	}

	// Loop over shocks
	for shockIdx, res := range boot {
		shockName := varNames[shockIdx]

		H, K := res.Point.Dims()

		for j := 0; j < K; j++ {
			respName := varNames[j]

			for h := 0; h < H; h++ {
				point := res.Point.At(h, j)
				low := res.Lower.At(h, j)
				high := res.Upper.At(h, j)

				record := []string{
					shockName,
					respName,
					fmt.Sprintf("%d", h),
					fmt.Sprintf("%f", point),
					fmt.Sprintf("%f", low),
					fmt.Sprintf("%f", high),
				}

				if err := writer.Write(record); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// OutputGrangerBootstrapMatrixToCSV writes the bootstrap GC results to CSV.
// Columns: CauseVar, EffectVar, FStatistic, AsymptoticP, BootPValue, Lags, Significant_Asymptotic, Significant_Bootstrap
func (rf *ReducedFormVAR) OutputGrangerBootstrapMatrixToCSV(
	path string,
	bootMat [][]*GrangerCausalityBootstrapResult,
	varNames []string,
) error {

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{
		"CauseVar",
		"EffectVar",
		"FStatistic",
		"AsymptoticP",
		"BootPValue",
		"Lags",
		"Significant_Asymptotic",
		"Significant_Bootstrap",
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	K := len(varNames)

	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			if i == j {
				continue
			}
			res := bootMat[i][j]
			if res == nil || res.Base == nil {
				continue
			}

			rec := []string{
				res.Base.CauseVar,
				res.Base.EffectVar,
				fmt.Sprintf("%f", res.Base.FStatistic),
				fmt.Sprintf("%f", res.Base.PValue),
				fmt.Sprintf("%f", res.BootPValue),
				fmt.Sprintf("%d", res.Base.Lags),
				fmt.Sprintf("%t", res.Base.Significant),
				fmt.Sprintf("%t", res.Significant),
			}

			if err := writer.Write(rec); err != nil {
				return err
			}
		}
	}

	return nil
}
func (rf *ReducedFormVAR) OutputIRFAnalysisToCSV(path string, analysis map[int][]float64, varNames []string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}

	defer file.Close()

	// Initialize a new CSV writer
	writer := csv.NewWriter(file)
	defer writer.Flush() // Ensure all buffered data is written

	// Write header
	header := []string{"Horizon"}
	for shockIdx := range analysis {
		var varName string
		if len(varNames) == len(analysis) {
			varName = varNames[shockIdx]
		} else {
			varName = fmt.Sprintf("Var%d", shockIdx+1)
		}
		header = append(header, "Shock_"+varName)
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	// Determine horizon from one of the analysis entries
	var horizon int
	for _, series := range analysis {
		horizon = len(series)
		break
	}

	// Write data rows
	for h := 0; h < horizon; h++ {
		record := []string{fmt.Sprintf("%d", h)}
		for shockIdx := range analysis {
			record = append(record, fmt.Sprintf("%f", analysis[shockIdx][h]))
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}
	return nil
}

func (rf *ReducedFormVAR) OutputForecastsToCSV(path string, fc *mat.Dense, varNames []string) error {

	rows, cols := fc.Dims()

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Initialize a new CSV writer
	writer := csv.NewWriter(file)
	defer writer.Flush() // Ensure all buffered data is written

	// Write header
	header := make([]string, cols)
	for j := 0; j < cols; j++ {
		if len(varNames) == cols {
			header[j] = varNames[j]
		} else {
			header[j] = fmt.Sprintf("Var%d", j+1)
		}
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	// Write data rows
	for i := 0; i < rows; i++ {
		record := make([]string, cols)
		for j := 0; j < cols; j++ {
			record[j] = fmt.Sprintf("%f", fc.At(i, j))
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}
	return nil
}
