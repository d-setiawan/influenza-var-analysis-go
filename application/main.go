// Authors: Rohan Adla, Arrio Gonsalves, Shreyan Nalwad, Dylan Setiawan
// Date: Dec 12th 2025
// Project: A VAR-based Computational Analysis of Influenza and Weather Dynamics
// Class: 02-613 at Caregie Mellon University

package main

import (
	"fmt"
	"os"
)

// This is the main function that runs the VAR analysis for the specified country and influenza type.
// The function expects two command-line arguments: the country name and the influenza type (A or B).
// The function will then perform the VAR analysis and output the results to CSV files.
// There are 14 steps in the whole process, including loading the CSV data, setting up the VAR specification,
// estimating the VAR model, forecasting, impulse response functions (IRFs), outputting the results to CSV files,
// and running additional analyses like Granger causality and bootstrap IRF analysis.

func main() {
	// expect 2 argument: country name, influenza type
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run main.go <country_name> <influenza_type>")
		return
	}
	country := os.Args[1]
	fmt.Println("Running VAR analysis for country:", country)

	// Determine filename based on country
	var filename string
	switch country {
	case "Singapore":
		filename = "Singapore/Training_Data_INF_"
	case "Qatar":
		filename = "Qatar/Training_Data_INF_"
	default:
		panic("Unsupported country: " + country + ". Options: Singapore, Qatar")
	}

	influenzaType := os.Args[2]
	var influenzaVarIndex string
	switch influenzaType {
	case "A":
		influenzaVarIndex = "A_transformed.csv"
	case "B":
		influenzaVarIndex = "B_transformed.csv"
	}

	// 1. Load CSV into TimeSeries
	ts, err := LoadCSVToTimeSeries("../Files/Final_Training_Data/" + filename + influenzaVarIndex)
	if err != nil {
		panic(err)
	}

	fmt.Println("Loaded series with", ts.Y.RawMatrix().Rows, "rows and",
		ts.Y.RawMatrix().Cols, "variables:", ts.VarNames)

	// 2. Set up VAR spec
	spec := ModelSpec{
		Lags:          6,
		Deterministic: DetConst, // or DetConstTrend, etc.
		HasExogenous:  false,
	}

	// 3. Estimate VAR
	rf, err := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
	if err != nil {
		panic(err)
	}

	rf.PrintCoefficients()

	// 4. Forecast 10 steps ahead
	fcst, err := rf.Forecast(ts.Y, 10)
	if err != nil {
		panic(err)
	}

	PrintForecast(fcst)

	// 5. IRF to shock sample variable 2
	irfMat, err := rf.IRF(12, 2)
	if err != nil {
		panic(err)
	}
	PrintIRF(irfMat, ts.VarNames, 2)

	// 6. Prints Summary
	rf.Summary(ts)

	// 7. Ouptput residuals to CSV
	err = rf.OutputForecastsToCSV("../Files/Output/forecast_results.csv", fcst, ts.VarNames)
	if err != nil {
		panic(err)
	}
	fmt.Println("Forecasts written to ../Files/Output/forecast_results.csv")

	// 8. Run Granger Causality Tests
	fmt.Println("Performing Granger Causality Analysis...")
	grangerResults, err := rf.GrangerCausalityMatrix(ts)
	if err != nil {
		panic(err)
	}
	PrintGrangerCausality(grangerResults, ts.VarNames)

	// 9. Output Granger results to CSV
	err = rf.OutputGrangerMatrixToCSV("../Files/Output/granger_results.csv", grangerResults, ts.VarNames)
	if err != nil {
		panic(err)
	}
	fmt.Println("Granger causality results written to ../Files/Output/granger_results.csv")

	// 10. Run varible shocking
	fmt.Println("Performing Variable Shocking Analysis...")
	// Print varibales and their indices
	for i, varName := range ts.VarNames {
		fmt.Printf("Variable %d: %s\n", i, varName)
	}
	// Run IRF analysis for shocks over 12 periods
	shockResults, err := rf.RunIRFAnalysis(0, 12)
	if err != nil {
		panic(err)
	}

	// 11. Output shocking results to CSV
	err = rf.OutputIRFAnalysisToCSV("../Files/Output/irf_results.csv", shockResults, ts.VarNames)
	if err != nil {
		panic(err)
	}

	fmt.Println("IRF analysis results written to ../Files/Output/irf_results.csv")

	// Bootstrap analysis is commented out as it takes a long time (upto 5 min) to run and is not necessary for the R interface.
	// Uncomment the following lines to run bootstrap analysis if needed.

	// 13. Run Bootstrap IRF Analysis
	// fmt.Println("      Running Bootstrap IRF Analysis     ")

	// bootOpts := BootstrapOptions{
	// 	NReplications: 500,   // increase to 1000+ for publication-quality bands
	// 	Horizon:       12,    // number of periods in IRF
	// 	Alpha:         0.05,  // 95% confidence interval
	// 	Seed:          12345, // or 0 to use current time
	// }

	// bootIRFs, err := rf.BootstrapIRF(ts, bootOpts)
	// if err != nil {
	// 	panic(fmt.Errorf("Bootstrap IRF failed: %v", err))
	// }

	// fmt.Println("Bootstrap IRF analysis completed.")
	// fmt.Printf("Computed IRFs with %d replications and horizon %d.\n",
	// 	bootOpts.NReplications, bootOpts.Horizon)

	// outPath := "../Files/Output/bootstrap_irf_results.csv"
	// err = OutputBootstrapIRFToCSV(outPath, bootIRFs, ts.VarNames)
	// if err != nil {
	// 	panic(fmt.Errorf("Failed to write bootstrap IRF CSV: %v", err))
	// }

	// fmt.Println("Bootstrap IRF results written to:", outPath)

	// // // 14. Run Bootstrap Granger Causality Analysis
	// fmt.Println("     Running Bootstrap Granger Causality      ")

	// gbOpts := GrangerBootstrapOptions{
	// 	NReplications: 500,   // bump to 1000+ if you want tighter p-values
	// 	Alpha:         0.05,  // 95% significance
	// 	Seed:          12345, // or 0 for time-based
	// }

	// bootGC, err := rf.BootstrapGrangerMatrix(ts, gbOpts)
	// if err != nil {
	// 	panic(err)
	// }

	// bootPath := "../Files/Output/granger_bootstrap_results.csv"
	// err = rf.OutputGrangerBootstrapMatrixToCSV(bootPath, bootGC, ts.VarNames)
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println("Bootstrap Granger causality results written to", bootPath)
}
