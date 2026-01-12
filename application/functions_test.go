// Authors: Rohan Adla, Arrio Gonsalves, Shreyan Nalwad, Dylan Setiawan
// Date: Dec 12th 2025
// Project: A VAR-based Computational Analysis of Influenza and Weather Dynamics
// Class: 02-613 at Caregie Mellon University

package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// almostEqual compares floats with tolerance
func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

// ReadDirectory reads all files in a directory
func ReadDirectory(directory string) []os.DirEntry {
	files, err := os.ReadDir(directory)
	if err != nil {
		panic(fmt.Sprintf("Error reading directory %s: %v", directory, err))
	}
	return files
}

// skipComments reads lines from scanner, skipping comment lines starting with #
func skipComments(scanner *bufio.Scanner) string {
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" && !strings.HasPrefix(line, "#") {
			return line
		}
	}
	return ""
}

// ============================================================================
// BOOTSTRAP QUANTILE TESTS
// ============================================================================

type BootstrapQuantileTest struct {
	Samples []float64
	Q       float64
	Result  float64
}

func ReadBootstrapQuantileTests(directory string) []BootstrapQuantileTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]BootstrapQuantileTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		samples, q := ReadBootstrapQuantileInput(directory + "input/" + inputFile.Name())
		tests[i].Samples = samples
		tests[i].Q = q
	}

	for i, outputFile := range outputFiles {
		result := ReadBootstrapQuantileOutput(directory + "output/" + outputFile.Name())
		tests[i].Result = result
	}

	return tests
}

func ReadBootstrapQuantileInput(file string) ([]float64, float64) {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	// Skip comment line
	line := skipComments(scanner)

	// First value is N (number of samples)
	n, err := strconv.Atoi(line)
	if err != nil {
		panic(fmt.Sprintf("Error parsing N: %v", err))
	}

	samples := make([]float64, n)
	for i := 0; i < n; i++ {
		line = skipComments(scanner)
		val, err := strconv.ParseFloat(line, 64)
		if err != nil {
			panic(fmt.Sprintf("Error parsing sample %d: %v", i, err))
		}
		samples[i] = val
	}

	// Last value is Q
	line = skipComments(scanner)
	q, err := strconv.ParseFloat(line, 64)
	if err != nil {
		panic(fmt.Sprintf("Error parsing Q: %v", err))
	}

	return samples, q
}

func ReadBootstrapQuantileOutput(file string) float64 {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	line := skipComments(scanner)
	result, err := strconv.ParseFloat(line, 64)
	if err != nil {
		panic(fmt.Sprintf("Error parsing result: %v", err))
	}

	return result
}

func TestBootstrapQuantile(t *testing.T) {
	tests := ReadBootstrapQuantileTests("Tests/BootstrapQuantile/")
	for i, test := range tests {
		got := bootstrapQuantile(test.Samples, test.Q)
		if !almostEqual(got, test.Result, 1e-6) {
			t.Errorf("Test %d: bootstrapQuantile(%v, %v) = %v; want %v",
				i+1, test.Samples, test.Q, got, test.Result)
		}
	}
}

// ============================================================================
// FORECAST TESTS
// ============================================================================

type ForecastTest struct {
	K       int
	Lags    int
	DetType Deterministic
	Steps   int
	A       []*mat.Dense
	C       *mat.Dense
	YHist   *mat.Dense
	Result  []float64
}

func ReadForecastTests(directory string) []ForecastTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]ForecastTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadForecastInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		tests[i].Result = ReadForecastOutput(directory + "output/" + outputFile.Name())
	}

	return tests
}

func ReadForecastInput(file string) ForecastTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	// Read K, lags, det_type, steps
	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	detTypeInt, _ := strconv.Atoi(line)
	detType := Deterministic(detTypeInt)

	line = skipComments(scanner)
	steps, _ := strconv.Atoi(line)

	// Read A matrices (lags * K*K values)
	A := make([]*mat.Dense, lags)
	for lag := 0; lag < lags; lag++ {
		data := make([]float64, K*K)
		for i := 0; i < K*K; i++ {
			line = skipComments(scanner)
			data[i], _ = strconv.ParseFloat(line, 64)
		}
		A[lag] = mat.NewDense(K, K, data)
	}

	// Read C matrix if deterministic
	var C *mat.Dense
	if detType == DetConst || detType == DetConstTrend {
		detCols := 0
		if detType == DetConst {
			detCols = 1
		} else if detType == DetConstTrend {
			detCols = 2
		}
		cData := make([]float64, K*detCols)
		for i := 0; i < K*detCols; i++ {
			line = skipComments(scanner)
			cData[i], _ = strconv.ParseFloat(line, 64)
		}
		C = mat.NewDense(K, detCols, cData)
	}

	// Read T (number of history rows)
	line = skipComments(scanner)
	T, _ := strconv.Atoi(line)

	// Read history data
	histData := make([]float64, T*K)
	for i := 0; i < T*K; i++ {
		line = skipComments(scanner)
		histData[i], _ = strconv.ParseFloat(line, 64)
	}
	YHist := mat.NewDense(T, K, histData)

	return ForecastTest{
		K:       K,
		Lags:    lags,
		DetType: detType,
		Steps:   steps,
		A:       A,
		C:       C,
		YHist:   YHist,
	}
}

func ReadForecastOutput(file string) []float64 {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	var results []float64

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		val, err := strconv.ParseFloat(line, 64)
		if err != nil {
			continue
		}
		results = append(results, val)
	}

	return results
}

func TestForecast(t *testing.T) {
	tests := ReadForecastTests("Tests/Forecast/")
	for i, test := range tests {
		spec := ModelSpec{
			Lags:          test.Lags,
			Deterministic: test.DetType,
			HasExogenous:  false,
		}

		rf := &ReducedFormVAR{
			Model: spec,
			A:     test.A,
			C:     test.C,
		}

		fcst, err := rf.Forecast(test.YHist, test.Steps)
		if err != nil {
			t.Errorf("Test %d: Forecast returned error: %v", i+1, err)
			continue
		}

		for j := 0; j < len(test.Result); j++ {
			got := fcst.At(j, 0)
			if !almostEqual(got, test.Result[j], 1e-4) {
				t.Errorf("Test %d: Forecast[%d] = %v, want %v", i+1, j, got, test.Result[j])
			}
		}
	}
}

// ============================================================================
// IRF TESTS
// ============================================================================

type IRFTest struct {
	K          int
	Lags       int
	DetType    Deterministic
	Horizon    int
	ShockIndex int
	A          []*mat.Dense
	SigmaU     *mat.SymDense
	Result     []float64
}

func ReadIRFTests(directory string) []IRFTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]IRFTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadIRFInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		tests[i].Result = ReadIRFOutput(directory + "output/" + outputFile.Name())
	}

	return tests
}

func ReadIRFInput(file string) IRFTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	// Read K, lags, det_type, horizon, shock_index
	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	detTypeInt, _ := strconv.Atoi(line)
	detType := Deterministic(detTypeInt)

	line = skipComments(scanner)
	horizon, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	shockIndex, _ := strconv.Atoi(line)

	// Read A matrices
	A := make([]*mat.Dense, lags)
	for lag := 0; lag < lags; lag++ {
		data := make([]float64, K*K)
		for i := 0; i < K*K; i++ {
			line = skipComments(scanner)
			data[i], _ = strconv.ParseFloat(line, 64)
		}
		A[lag] = mat.NewDense(K, K, data)
	}

	// Read SigmaU (K*K values)
	sigmaData := make([]float64, K*K)
	for i := 0; i < K*K; i++ {
		line = skipComments(scanner)
		sigmaData[i], _ = strconv.ParseFloat(line, 64)
	}
	SigmaU := mat.NewSymDense(K, sigmaData)

	return IRFTest{
		K:          K,
		Lags:       lags,
		DetType:    detType,
		Horizon:    horizon,
		ShockIndex: shockIndex,
		A:          A,
		SigmaU:     SigmaU,
	}
}

func ReadIRFOutput(file string) []float64 {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	var results []float64

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		val, err := strconv.ParseFloat(line, 64)
		if err != nil {
			continue
		}
		results = append(results, val)
	}

	return results
}

func TestIRF(t *testing.T) {
	tests := ReadIRFTests("Tests/IRF/")
	for i, test := range tests {
		spec := ModelSpec{
			Lags:          test.Lags,
			Deterministic: test.DetType,
			HasExogenous:  false,
		}

		rf := &ReducedFormVAR{
			Model:  spec,
			A:      test.A,
			SigmaU: test.SigmaU,
		}

		irf, err := rf.IRF(test.Horizon, test.ShockIndex)
		if err != nil {
			t.Errorf("Test %d: IRF returned error: %v", i+1, err)
			continue
		}

		for h := 0; h < len(test.Result); h++ {
			got := irf.At(h, 0)
			if !almostEqual(got, test.Result[h], 1e-4) {
				t.Errorf("Test %d: IRF[%d] = %v, want %v", i+1, h, got, test.Result[h])
			}
		}
	}
}

// ============================================================================
// SPEC TESTS
// ============================================================================

type SpecTest struct {
	Lags         int
	DetType      Deterministic
	HasExogenous bool
}

func ReadSpecTests(directory string) []SpecTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]SpecTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadSpecInput(directory + "input/" + inputFile.Name())
	}

	// Output should match input for Spec getter
	return tests
}

func ReadSpecInput(file string) SpecTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	detTypeInt, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	hasExogInt, _ := strconv.Atoi(line)

	return SpecTest{
		Lags:         lags,
		DetType:      Deterministic(detTypeInt),
		HasExogenous: hasExogInt == 1,
	}
}

func TestSpec(t *testing.T) {
	tests := ReadSpecTests("Tests/Spec/")
	for i, test := range tests {
		spec := ModelSpec{
			Lags:          test.Lags,
			Deterministic: test.DetType,
			HasExogenous:  test.HasExogenous,
		}
		rf := &ReducedFormVAR{Model: spec}

		got := rf.Spec()
		if got.Lags != test.Lags {
			t.Errorf("Test %d: Spec().Lags = %d, want %d", i+1, got.Lags, test.Lags)
		}
		if got.Deterministic != test.DetType {
			t.Errorf("Test %d: Spec().Deterministic = %v, want %v", i+1, got.Deterministic, test.DetType)
		}
		if got.HasExogenous != test.HasExogenous {
			t.Errorf("Test %d: Spec().HasExogenous = %v, want %v", i+1, got.HasExogenous, test.HasExogenous)
		}
	}
}

// ============================================================================
// PHI TESTS
// ============================================================================

type PhiTest struct {
	K       int
	NumLags int
	A       []*mat.Dense
	ExpLen  int
	First00 float64
}

func ReadPhiTests(directory string) []PhiTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]PhiTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadPhiInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		expLen, first00 := ReadPhiOutput(directory + "output/" + outputFile.Name())
		tests[i].ExpLen = expLen
		tests[i].First00 = first00
	}

	return tests
}

func ReadPhiInput(file string) PhiTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	numLags, _ := strconv.Atoi(line)

	A := make([]*mat.Dense, numLags)
	for lag := 0; lag < numLags; lag++ {
		data := make([]float64, K*K)
		idx := 0
		for row := 0; row < K; row++ {
			line = skipComments(scanner)
			parts := strings.Fields(line)
			for _, p := range parts {
				data[idx], _ = strconv.ParseFloat(p, 64)
				idx++
			}
		}
		A[lag] = mat.NewDense(K, K, data)
	}

	return PhiTest{K: K, NumLags: numLags, A: A}
}

func ReadPhiOutput(file string) (int, float64) {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	line := skipComments(scanner)

	// Try to parse as expLen first
	expLen, err := strconv.Atoi(line)
	if err != nil {
		// Single value case
		first00, _ := strconv.ParseFloat(line, 64)
		return 1, first00
	}

	line = skipComments(scanner)
	first00, _ := strconv.ParseFloat(line, 64)

	return expLen, first00
}

func TestPhi(t *testing.T) {
	tests := ReadPhiTests("Tests/Phi/")
	for i, test := range tests {
		rf := &ReducedFormVAR{A: test.A}

		phi := rf.Phi()
		if len(phi) != test.ExpLen {
			t.Errorf("Test %d: len(Phi()) = %d, want %d", i+1, len(phi), test.ExpLen)
		}
		if phi[0].At(0, 0) != test.First00 {
			t.Errorf("Test %d: Phi()[0].At(0,0) = %v, want %v", i+1, phi[0].At(0, 0), test.First00)
		}
	}
}

// ============================================================================
// COVU TESTS
// ============================================================================

type CovUTest struct {
	K      int
	SigmaU *mat.SymDense
	Exp00  float64
	Exp01  float64
}

func ReadCovUTests(directory string) []CovUTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]CovUTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadCovUInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		exp00, exp01 := ReadCovUOutput(directory + "output/" + outputFile.Name())
		tests[i].Exp00 = exp00
		tests[i].Exp01 = exp01
	}

	return tests
}

func ReadCovUInput(file string) CovUTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	data := make([]float64, K*K)
	idx := 0
	for row := 0; row < K; row++ {
		line = skipComments(scanner)
		parts := strings.Fields(line)
		for _, p := range parts {
			data[idx], _ = strconv.ParseFloat(p, 64)
			idx++
		}
	}

	return CovUTest{K: K, SigmaU: mat.NewSymDense(K, data)}
}

func ReadCovUOutput(file string) (float64, float64) {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	exp00, _ := strconv.ParseFloat(line, 64)

	line = skipComments(scanner)
	if line == "" {
		return exp00, 0.0
	}
	exp01, _ := strconv.ParseFloat(line, 64)

	return exp00, exp01
}

func TestCovU(t *testing.T) {
	tests := ReadCovUTests("Tests/CovU/")
	for i, test := range tests {
		rf := &ReducedFormVAR{SigmaU: test.SigmaU}

		got := rf.CovU()
		if got.At(0, 0) != test.Exp00 {
			t.Errorf("Test %d: CovU().At(0,0) = %v, want %v", i+1, got.At(0, 0), test.Exp00)
		}
		if test.K > 1 && got.At(0, 1) != test.Exp01 {
			t.Errorf("Test %d: CovU().At(0,1) = %v, want %v", i+1, got.At(0, 1), test.Exp01)
		}
	}
}

// ============================================================================
// ESTIMATE TESTS
// ============================================================================

type EstimateTest struct {
	K        int
	T        int
	Lags     int
	DetType  Deterministic
	Y        *mat.Dense
	ExpPhi00 float64
}

func ReadEstimateTests(directory string) []EstimateTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]EstimateTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadEstimateInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		tests[i].ExpPhi00 = ReadEstimateOutput(directory + "output/" + outputFile.Name())
	}

	return tests
}

func ReadEstimateInput(file string) EstimateTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	T, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	detTypeInt, _ := strconv.Atoi(line)

	data := make([]float64, T*K)
	for i := 0; i < T*K; i++ {
		line = skipComments(scanner)
		data[i], _ = strconv.ParseFloat(line, 64)
	}

	return EstimateTest{
		K:       K,
		T:       T,
		Lags:    lags,
		DetType: Deterministic(detTypeInt),
		Y:       mat.NewDense(T, K, data),
	}
}

func ReadEstimateOutput(file string) float64 {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	line := skipComments(scanner)
	result, _ := strconv.ParseFloat(line, 64)

	return result
}

func TestEstimate(t *testing.T) {
	tests := ReadEstimateTests("Tests/Estimate/")
	for i, test := range tests {
		ts := &TimeSeries{
			Y:        test.Y,
			VarNames: []string{"y"},
		}

		spec := ModelSpec{
			Lags:          test.Lags,
			Deterministic: test.DetType,
			HasExogenous:  false,
		}

		est := &OLSEstimator{}
		rf, err := est.Estimate(ts, spec, EstimationOptions{})
		if err != nil {
			t.Errorf("Test %d: Estimate returned error: %v", i+1, err)
			continue
		}

		if len(rf.A) < 1 {
			t.Errorf("Test %d: len(rf.A) = %d, want >= 1", i+1, len(rf.A))
			continue
		}

		phiHat := rf.A[0].At(0, 0)
		if !almostEqual(phiHat, test.ExpPhi00, 1e-2) {
			t.Errorf("Test %d: Estimated phi = %v, want approx %v", i+1, phiHat, test.ExpPhi00)
		}
	}
}

// ============================================================================
// COMPUTE RESIDUALS TESTS
// ============================================================================

type ComputeResidualsTest struct {
	K          int
	T          int
	Lags       int
	DetType    Deterministic
	A          []*mat.Dense
	Y          *mat.Dense
	ExpNumRows int
	ExpResids  []float64
}

func ReadComputeResidualsTests(directory string) []ComputeResidualsTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]ComputeResidualsTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadComputeResidualsInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		expNumRows, expResids := ReadComputeResidualsOutput(directory + "output/" + outputFile.Name())
		tests[i].ExpNumRows = expNumRows
		tests[i].ExpResids = expResids
	}

	return tests
}

func ReadComputeResidualsInput(file string) ComputeResidualsTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	T, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	detTypeInt, _ := strconv.Atoi(line)

	// Read A matrices
	A := make([]*mat.Dense, lags)
	for lag := 0; lag < lags; lag++ {
		data := make([]float64, K*K)
		for j := 0; j < K*K; j++ {
			line = skipComments(scanner)
			data[j], _ = strconv.ParseFloat(line, 64)
		}
		A[lag] = mat.NewDense(K, K, data)
	}

	// Read Y data
	data := make([]float64, T*K)
	for i := 0; i < T*K; i++ {
		line = skipComments(scanner)
		data[i], _ = strconv.ParseFloat(line, 64)
	}

	return ComputeResidualsTest{
		K:       K,
		T:       T,
		Lags:    lags,
		DetType: Deterministic(detTypeInt),
		A:       A,
		Y:       mat.NewDense(T, K, data),
	}
}

func ReadComputeResidualsOutput(file string) (int, []float64) {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	expNumRows, _ := strconv.Atoi(line)

	resids := make([]float64, expNumRows)
	for i := 0; i < expNumRows; i++ {
		line = skipComments(scanner)
		resids[i], _ = strconv.ParseFloat(line, 64)
	}

	return expNumRows, resids
}

func TestComputeResiduals(t *testing.T) {
	tests := ReadComputeResidualsTests("Tests/ComputeResiduals/")
	for i, test := range tests {
		ts := &TimeSeries{
			Y:        test.Y,
			VarNames: []string{"y"},
		}

		spec := ModelSpec{
			Lags:          test.Lags,
			Deterministic: test.DetType,
		}

		rf := &ReducedFormVAR{
			Model: spec,
			A:     test.A,
		}

		resU, err := rf.computeResiduals(ts)
		if err != nil {
			t.Errorf("Test %d: computeResiduals returned error: %v", i+1, err)
			continue
		}

		rows, _ := resU.Dims()
		if rows != test.ExpNumRows {
			t.Errorf("Test %d: residuals rows = %d, want %d", i+1, rows, test.ExpNumRows)
			continue
		}

		for j := 0; j < test.ExpNumRows; j++ {
			if !almostEqual(resU.At(j, 0), test.ExpResids[j], 1e-4) {
				t.Errorf("Test %d: residual[%d] = %v, want %v", i+1, j, resU.At(j, 0), test.ExpResids[j])
			}
		}
	}
}

// ============================================================================
// GRANGER CAUSALITY TESTS
// ============================================================================

type GrangerCausalityTest struct {
	K           int
	T           int
	Lags        int
	DetType     Deterministic
	CauseIdx    int
	EffectIdx   int
	Y           *mat.Dense
	ExpCauseVar string
	ExpEffVar   string
	ExpLags     int
	ExpError    bool
}

func ReadGrangerCausalityTests(directory string) []GrangerCausalityTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]GrangerCausalityTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadGrangerCausalityInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		causeVar, effVar, lags, expError := ReadGrangerCausalityOutput(directory + "output/" + outputFile.Name())
		tests[i].ExpCauseVar = causeVar
		tests[i].ExpEffVar = effVar
		tests[i].ExpLags = lags
		tests[i].ExpError = expError
	}

	return tests
}

func ReadGrangerCausalityInput(file string) GrangerCausalityTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	T, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	detTypeInt, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	causeIdx, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	effectIdx, _ := strconv.Atoi(line)

	// Read Y data (T x K, space-separated per row)
	data := make([]float64, T*K)
	idx := 0
	for row := 0; row < T; row++ {
		line = skipComments(scanner)
		parts := strings.Fields(line)
		for _, p := range parts {
			val, _ := strconv.ParseFloat(p, 64)
			data[idx] = val
			idx++
		}
	}

	return GrangerCausalityTest{
		K:         K,
		T:         T,
		Lags:      lags,
		DetType:   Deterministic(detTypeInt),
		CauseIdx:  causeIdx,
		EffectIdx: effectIdx,
		Y:         mat.NewDense(T, K, data),
	}
}

func ReadGrangerCausalityOutput(file string) (string, string, int, bool) {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	if line == "error" {
		return "", "", 0, true
	}

	causeVar := line

	line = skipComments(scanner)
	effVar := line

	line = skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	return causeVar, effVar, lags, false
}

func TestGrangerCausality(t *testing.T) {
	tests := ReadGrangerCausalityTests("Tests/GrangerCausality/")
	for i, test := range tests {
		// Create variable names
		varNames := make([]string, test.K)
		for j := 0; j < test.K; j++ {
			if test.K == 2 {
				varNames[j] = fmt.Sprintf("y%d", j+1)
			} else {
				varNames[j] = fmt.Sprintf("var%d", j)
			}
		}

		ts := &TimeSeries{
			Y:        test.Y,
			VarNames: varNames,
		}

		spec := ModelSpec{
			Lags:          test.Lags,
			Deterministic: test.DetType,
			HasExogenous:  false,
		}

		// First estimate the VAR
		rf, err := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
		if err != nil {
			t.Errorf("Test %d: Estimate failed: %v", i+1, err)
			continue
		}

		// Test Granger causality
		result, err := rf.GrangerCausality(ts, test.CauseIdx, test.EffectIdx)

		if test.ExpError {
			if err == nil {
				t.Errorf("Test %d: Expected error but got none", i+1)
			}
			continue
		}

		if err != nil {
			t.Errorf("Test %d: GrangerCausality returned error: %v", i+1, err)
			continue
		}

		if result.CauseVar != test.ExpCauseVar {
			t.Errorf("Test %d: CauseVar = %s, want %s", i+1, result.CauseVar, test.ExpCauseVar)
		}
		if result.EffectVar != test.ExpEffVar {
			t.Errorf("Test %d: EffectVar = %s, want %s", i+1, result.EffectVar, test.ExpEffVar)
		}
		if result.Lags != test.ExpLags {
			t.Errorf("Test %d: Lags = %d, want %d", i+1, result.Lags, test.ExpLags)
		}

		// F-statistic and p-value should be valid numbers
		if math.IsNaN(result.FStatistic) || math.IsInf(result.FStatistic, 0) {
			t.Errorf("Test %d: FStatistic is NaN or Inf: %v", i+1, result.FStatistic)
		}
		if result.PValue < 0 || result.PValue > 1 {
			t.Errorf("Test %d: PValue out of range: %v", i+1, result.PValue)
		}
	}
}

// ============================================================================
// RUN IRF ANALYSIS TESTS
// ============================================================================

type RunIRFAnalysisTest struct {
	K         int
	Lags      int
	DetType   Deterministic
	VarIndex  int
	Horizon   int
	A         []*mat.Dense
	SigmaU    *mat.SymDense
	ExpShocks map[int][]float64
}

func ReadRunIRFAnalysisTests(directory string) []RunIRFAnalysisTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]RunIRFAnalysisTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadRunIRFAnalysisInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		tests[i].ExpShocks = ReadRunIRFAnalysisOutput(directory + "output/" + outputFile.Name())
	}

	return tests
}

func ReadRunIRFAnalysisInput(file string) RunIRFAnalysisTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	detTypeInt, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	varIndex, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	horizon, _ := strconv.Atoi(line)

	// Read A matrices
	A := make([]*mat.Dense, lags)
	for lag := 0; lag < lags; lag++ {
		data := make([]float64, K*K)
		idx := 0
		for row := 0; row < K; row++ {
			line = skipComments(scanner)
			parts := strings.Fields(line)
			for _, p := range parts {
				data[idx], _ = strconv.ParseFloat(p, 64)
				idx++
			}
		}
		A[lag] = mat.NewDense(K, K, data)
	}

	// Read SigmaU
	sigmaData := make([]float64, K*K)
	idx := 0
	for row := 0; row < K; row++ {
		line = skipComments(scanner)
		parts := strings.Fields(line)
		for _, p := range parts {
			sigmaData[idx], _ = strconv.ParseFloat(p, 64)
			idx++
		}
	}

	return RunIRFAnalysisTest{
		K:        K,
		Lags:     lags,
		DetType:  Deterministic(detTypeInt),
		VarIndex: varIndex,
		Horizon:  horizon,
		A:        A,
		SigmaU:   mat.NewSymDense(K, sigmaData),
	}
}

func ReadRunIRFAnalysisOutput(file string) map[int][]float64 {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	result := make(map[int][]float64)

	line := skipComments(scanner)
	numShocks, _ := strconv.Atoi(line)

	for i := 0; i < numShocks; i++ {
		line = skipComments(scanner)
		parts := strings.Fields(line)
		shockIdx, _ := strconv.Atoi(parts[0])
		horizon, _ := strconv.Atoi(parts[1])
		values := make([]float64, horizon)
		for j := 0; j < horizon; j++ {
			values[j], _ = strconv.ParseFloat(parts[2+j], 64)
		}
		result[shockIdx] = values
	}

	return result
}

func TestRunIRFAnalysis(t *testing.T) {
	tests := ReadRunIRFAnalysisTests("Tests/RunIRFAnalysis/")
	for i, test := range tests {
		spec := ModelSpec{
			Lags:          test.Lags,
			Deterministic: test.DetType,
			HasExogenous:  false,
		}

		rf := &ReducedFormVAR{
			Model:  spec,
			A:      test.A,
			SigmaU: test.SigmaU,
		}

		results, err := rf.RunIRFAnalysis(test.VarIndex, test.Horizon)
		if err != nil {
			t.Errorf("Test %d: RunIRFAnalysis returned error: %v", i+1, err)
			continue
		}

		if len(results) != len(test.ExpShocks) {
			t.Errorf("Test %d: len(results) = %d, want %d", i+1, len(results), len(test.ExpShocks))
			continue
		}

		for shockIdx, expValues := range test.ExpShocks {
			gotValues, ok := results[shockIdx]
			if !ok {
				t.Errorf("Test %d: missing shock %d in results", i+1, shockIdx)
				continue
			}
			if len(gotValues) != len(expValues) {
				t.Errorf("Test %d: shock %d len = %d, want %d", i+1, shockIdx, len(gotValues), len(expValues))
				continue
			}
			for h, exp := range expValues {
				if !almostEqual(gotValues[h], exp, 1e-2) {
					t.Errorf("Test %d: shock %d horizon %d = %v, want %v", i+1, shockIdx, h, gotValues[h], exp)
				}
			}
		}
	}
}

// ============================================================================
// GRANGER CAUSALITY MATRIX TESTS
// ============================================================================

type GrangerCausalityMatrixTest struct {
	K          int
	T          int
	Lags       int
	DetType    Deterministic
	Y          *mat.Dense
	ExpResults []struct{ CauseIdx, EffectIdx, Lags int }
}

func ReadGrangerCausalityMatrixTests(directory string) []GrangerCausalityMatrixTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]GrangerCausalityMatrixTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadGrangerCausalityMatrixInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		tests[i].ExpResults = ReadGrangerCausalityMatrixOutput(directory + "output/" + outputFile.Name())
	}

	return tests
}

func ReadGrangerCausalityMatrixInput(file string) GrangerCausalityMatrixTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	T, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	detTypeInt, _ := strconv.Atoi(line)

	// Read Y data
	data := make([]float64, T*K)
	idx := 0
	for row := 0; row < T; row++ {
		line = skipComments(scanner)
		parts := strings.Fields(line)
		for _, p := range parts {
			data[idx], _ = strconv.ParseFloat(p, 64)
			idx++
		}
	}

	return GrangerCausalityMatrixTest{
		K:       K,
		T:       T,
		Lags:    lags,
		DetType: Deterministic(detTypeInt),
		Y:       mat.NewDense(T, K, data),
	}
}

func ReadGrangerCausalityMatrixOutput(file string) []struct{ CauseIdx, EffectIdx, Lags int } {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	parts := strings.Fields(line)
	numResults, _ := strconv.Atoi(parts[1])

	results := make([]struct{ CauseIdx, EffectIdx, Lags int }, numResults)
	for i := 0; i < numResults; i++ {
		line = skipComments(scanner)
		parts = strings.Fields(line)
		results[i].CauseIdx, _ = strconv.Atoi(parts[0])
		results[i].EffectIdx, _ = strconv.Atoi(parts[1])
		results[i].Lags, _ = strconv.Atoi(parts[2])
	}

	return results
}

func TestGrangerCausalityMatrix(t *testing.T) {
	tests := ReadGrangerCausalityMatrixTests("Tests/GrangerCausalityMatrix/")
	for i, test := range tests {
		varNames := make([]string, test.K)
		for j := 0; j < test.K; j++ {
			varNames[j] = fmt.Sprintf("var%d", j)
		}

		ts := &TimeSeries{
			Y:        test.Y,
			VarNames: varNames,
		}

		spec := ModelSpec{
			Lags:          test.Lags,
			Deterministic: test.DetType,
		}

		rf, err := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
		if err != nil {
			t.Errorf("Test %d: Estimate failed: %v", i+1, err)
			continue
		}

		matrix, err := rf.GrangerCausalityMatrix(ts)
		if err != nil {
			t.Errorf("Test %d: GrangerCausalityMatrix returned error: %v", i+1, err)
			continue
		}

		if len(matrix) != test.K {
			t.Errorf("Test %d: len(matrix) = %d, want %d", i+1, len(matrix), test.K)
			continue
		}

		// Check diagonal is nil
		for j := 0; j < test.K; j++ {
			if matrix[j][j] != nil {
				t.Errorf("Test %d: matrix[%d][%d] should be nil", i+1, j, j)
			}
		}

		// Verify expected results exist
		for _, exp := range test.ExpResults {
			res := matrix[exp.CauseIdx][exp.EffectIdx]
			if res == nil {
				t.Errorf("Test %d: matrix[%d][%d] should not be nil", i+1, exp.CauseIdx, exp.EffectIdx)
				continue
			}
			if res.Lags != exp.Lags {
				t.Errorf("Test %d: matrix[%d][%d].Lags = %d, want %d", i+1, exp.CauseIdx, exp.EffectIdx, res.Lags, exp.Lags)
			}
		}
	}
}

func TestOutputForecastsToCSV(t *testing.T) {
	tmpFile := "test_forecasts.csv"
	defer os.Remove(tmpFile)

	fc := mat.NewDense(3, 2, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	varNames := []string{"var1", "var2"}

	rf := &ReducedFormVAR{}
	err := rf.OutputForecastsToCSV(tmpFile, fc, varNames)
	if err != nil {
		t.Fatalf("OutputForecastsToCSV returned error: %v", err)
	}

	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Error("Output file was not created")
	}
}

func TestOutputIRFAnalysisToCSV(t *testing.T) {
	tmpFile := "test_irf_analysis.csv"
	defer os.Remove(tmpFile)

	analysis := map[int][]float64{
		0: {1.0, 0.5, 0.25},
		1: {0.0, 0.1, 0.2},
	}
	varNames := []string{"x", "y"}

	rf := &ReducedFormVAR{}
	err := rf.OutputIRFAnalysisToCSV(tmpFile, analysis, varNames)
	if err != nil {
		t.Fatalf("OutputIRFAnalysisToCSV returned error: %v", err)
	}

	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Error("Output file was not created")
	}
}

func TestOutputGrangerMatrixToCSV(t *testing.T) {
	tmpFile := "test_granger_matrix.csv"
	defer os.Remove(tmpFile)

	varNames := []string{"a", "b"}
	gcMatrix := [][]*GrangerCausalityResult{
		{nil, {CauseVar: "a", EffectVar: "b", FStatistic: 2.5, PValue: 0.1, Lags: 1, Significant: false}},
		{{CauseVar: "b", EffectVar: "a", FStatistic: 5.0, PValue: 0.02, Lags: 1, Significant: true}, nil},
	}

	rf := &ReducedFormVAR{}
	err := rf.OutputGrangerMatrixToCSV(tmpFile, gcMatrix, varNames)
	if err != nil {
		t.Fatalf("OutputGrangerMatrixToCSV returned error: %v", err)
	}

	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Error("Output file was not created")
	}
}

// ============================================================================
// SIMULATE BOOTSTRAP SERIES TESTS
// ============================================================================

type SimulateBootstrapSeriesTest struct {
	K            int
	T            int
	Lags         int
	DetType      Deterministic
	Seed         int64
	A            []*mat.Dense
	C            *mat.Dense
	ResU         *mat.Dense
	Y            *mat.Dense
	ExpT         int
	ExpK         int
	ExpFirstRows [][]float64
}

func ReadSimulateBootstrapSeriesTests(directory string) []SimulateBootstrapSeriesTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]SimulateBootstrapSeriesTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadSimulateBootstrapSeriesInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		expT, expK, firstRows := ReadSimulateBootstrapSeriesOutput(directory + "output/" + outputFile.Name())
		tests[i].ExpT = expT
		tests[i].ExpK = expK
		tests[i].ExpFirstRows = firstRows
	}

	return tests
}

func ReadSimulateBootstrapSeriesInput(file string) SimulateBootstrapSeriesTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	T, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	detTypeInt, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	seed, _ := strconv.ParseInt(line, 10, 64)

	// Read A matrices
	A := make([]*mat.Dense, lags)
	for lag := 0; lag < lags; lag++ {
		data := make([]float64, K*K)
		idx := 0
		for row := 0; row < K; row++ {
			line = skipComments(scanner)
			parts := strings.Fields(line)
			for _, p := range parts {
				data[idx], _ = strconv.ParseFloat(p, 64)
				idx++
			}
		}
		A[lag] = mat.NewDense(K, K, data)
	}

	// Read C if deterministic
	detType := Deterministic(detTypeInt)
	var C *mat.Dense
	detCols := 0
	if detType == DetConst {
		detCols = 1
	} else if detType == DetConstTrend {
		detCols = 2
	} else if detType == DetTrend {
		detCols = 1
	}
	if detCols > 0 {
		cData := make([]float64, K*detCols)
		for i := 0; i < K; i++ {
			line = skipComments(scanner)
			cData[i], _ = strconv.ParseFloat(line, 64)
		}
		C = mat.NewDense(K, detCols, cData)
	}

	// Read residuals ((T-lags) x K)
	Treg := T - lags
	resData := make([]float64, Treg*K)
	idx := 0
	for row := 0; row < Treg; row++ {
		line = skipComments(scanner)
		parts := strings.Fields(line)
		for _, p := range parts {
			resData[idx], _ = strconv.ParseFloat(p, 64)
			idx++
		}
	}
	ResU := mat.NewDense(Treg, K, resData)

	// Read Y (T x K)
	yData := make([]float64, T*K)
	idx = 0
	for row := 0; row < T; row++ {
		line = skipComments(scanner)
		parts := strings.Fields(line)
		for _, p := range parts {
			yData[idx], _ = strconv.ParseFloat(p, 64)
			idx++
		}
	}
	Y := mat.NewDense(T, K, yData)

	return SimulateBootstrapSeriesTest{
		K:       K,
		T:       T,
		Lags:    lags,
		DetType: detType,
		Seed:    seed,
		A:       A,
		C:       C,
		ResU:    ResU,
		Y:       Y,
	}
}

func ReadSimulateBootstrapSeriesOutput(file string) (int, int, [][]float64) {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	parts := strings.Fields(line)
	expT, _ := strconv.Atoi(parts[0])
	expK, _ := strconv.Atoi(parts[1])

	// Read first rows that should match
	var firstRows [][]float64
	for scanner.Scan() {
		line = strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts = strings.Fields(line)
		row := make([]float64, len(parts))
		for i, p := range parts {
			row[i], _ = strconv.ParseFloat(p, 64)
		}
		firstRows = append(firstRows, row)
	}

	return expT, expK, firstRows
}

func TestSimulateBootstrapSeries(t *testing.T) {
	tests := ReadSimulateBootstrapSeriesTests("Tests/SimulateBootstrapSeries/")
	for i, test := range tests {
		varNames := make([]string, test.K)
		for j := 0; j < test.K; j++ {
			varNames[j] = fmt.Sprintf("var%d", j)
		}

		ts := &TimeSeries{
			Y:        test.Y,
			VarNames: varNames,
		}

		spec := ModelSpec{
			Lags:          test.Lags,
			Deterministic: test.DetType,
		}

		rf := &ReducedFormVAR{
			Model: spec,
			A:     test.A,
			C:     test.C,
		}

		rng := rand.New(rand.NewSource(test.Seed))

		tsStar, err := rf.simulateBootstrapSeries(ts, test.ResU, rng)
		if err != nil {
			t.Errorf("Test %d: simulateBootstrapSeries returned error: %v", i+1, err)
			continue
		}

		rows, cols := tsStar.Y.Dims()
		if rows != test.ExpT {
			t.Errorf("Test %d: bootstrap Y rows = %d, want %d", i+1, rows, test.ExpT)
		}
		if cols != test.ExpK {
			t.Errorf("Test %d: bootstrap Y cols = %d, want %d", i+1, cols, test.ExpK)
		}

		// Verify first p rows match original
		for r, expRow := range test.ExpFirstRows {
			for c, expVal := range expRow {
				if !almostEqual(tsStar.Y.At(r, c), expVal, 1e-6) {
					t.Errorf("Test %d: bootstrap Y[%d][%d] = %v, want %v",
						i+1, r, c, tsStar.Y.At(r, c), expVal)
				}
			}
		}
	}
}

// ============================================================================
// BOOTSTRAP IRF TESTS
// ============================================================================

type BootstrapIRFTest struct {
	K             int
	T             int
	Lags          int
	DetType       Deterministic
	NReplications int
	Horizon       int
	Alpha         float64
	Seed          int64
	Y             *mat.Dense
	ExpShocks     []struct{ ShockIdx, Horizon int }
}

func ReadBootstrapIRFTests(directory string) []BootstrapIRFTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]BootstrapIRFTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadBootstrapIRFInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		tests[i].ExpShocks = ReadBootstrapIRFOutput(directory + "output/" + outputFile.Name())
	}

	return tests
}

func ReadBootstrapIRFInput(file string) BootstrapIRFTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	T, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	detTypeInt, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	nRep, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	horizon, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	alpha, _ := strconv.ParseFloat(line, 64)

	line = skipComments(scanner)
	seed, _ := strconv.ParseInt(line, 10, 64)

	// Read Y data
	data := make([]float64, T*K)
	idx := 0
	for row := 0; row < T; row++ {
		line = skipComments(scanner)
		parts := strings.Fields(line)
		for _, p := range parts {
			data[idx], _ = strconv.ParseFloat(p, 64)
			idx++
		}
	}

	return BootstrapIRFTest{
		K:             K,
		T:             T,
		Lags:          lags,
		DetType:       Deterministic(detTypeInt),
		NReplications: nRep,
		Horizon:       horizon,
		Alpha:         alpha,
		Seed:          seed,
		Y:             mat.NewDense(T, K, data),
	}
}

func ReadBootstrapIRFOutput(file string) []struct{ ShockIdx, Horizon int } {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	numShocks, _ := strconv.Atoi(line)

	results := make([]struct{ ShockIdx, Horizon int }, numShocks)
	for i := 0; i < numShocks; i++ {
		line = skipComments(scanner)
		parts := strings.Fields(line)
		results[i].ShockIdx, _ = strconv.Atoi(parts[0])
		results[i].Horizon, _ = strconv.Atoi(parts[1])
	}

	return results
}

func TestBootstrapIRF(t *testing.T) {
	tests := ReadBootstrapIRFTests("Tests/BootstrapIRF/")
	for i, test := range tests {
		varNames := make([]string, test.K)
		for j := 0; j < test.K; j++ {
			varNames[j] = fmt.Sprintf("var%d", j)
		}

		ts := &TimeSeries{
			Y:        test.Y,
			VarNames: varNames,
		}

		spec := ModelSpec{
			Lags:          test.Lags,
			Deterministic: test.DetType,
		}

		rf, err := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
		if err != nil {
			t.Errorf("Test %d: Estimate failed: %v", i+1, err)
			continue
		}

		opts := BootstrapOptions{
			NReplications: test.NReplications,
			Horizon:       test.Horizon,
			Alpha:         test.Alpha,
			Seed:          test.Seed,
		}

		results, err := rf.BootstrapIRF(ts, opts)
		if err != nil {
			t.Errorf("Test %d: BootstrapIRF returned error: %v", i+1, err)
			continue
		}

		if len(results) != len(test.ExpShocks) {
			t.Errorf("Test %d: len(results) = %d, want %d", i+1, len(results), len(test.ExpShocks))
			continue
		}

		for _, exp := range test.ExpShocks {
			res, ok := results[exp.ShockIdx]
			if !ok {
				t.Errorf("Test %d: missing shock %d in results", i+1, exp.ShockIdx)
				continue
			}
			if res.ShockIndex != exp.ShockIdx {
				t.Errorf("Test %d: ShockIndex = %d, want %d", i+1, res.ShockIndex, exp.ShockIdx)
			}
			if res.Horizon != exp.Horizon {
				t.Errorf("Test %d: Horizon = %d, want %d", i+1, res.Horizon, exp.Horizon)
			}
			if res.Point == nil {
				t.Errorf("Test %d: Point IRF is nil for shock %d", i+1, exp.ShockIdx)
			}
			if res.Lower == nil {
				t.Errorf("Test %d: Lower CI is nil for shock %d", i+1, exp.ShockIdx)
			}
			if res.Upper == nil {
				t.Errorf("Test %d: Upper CI is nil for shock %d", i+1, exp.ShockIdx)
			}
		}
	}
}

// ============================================================================
// BOOTSTRAP GRANGER MATRIX TESTS
// ============================================================================

type BootstrapGrangerMatrixTest struct {
	K             int
	T             int
	Lags          int
	DetType       Deterministic
	NReplications int
	Alpha         float64
	Seed          int64
	Y             *mat.Dense
	ExpResults    []struct{ CauseIdx, EffectIdx int }
}

func ReadBootstrapGrangerMatrixTests(directory string) []BootstrapGrangerMatrixTest {
	inputFiles := ReadDirectory(directory + "input")
	outputFiles := ReadDirectory(directory + "output")

	if len(inputFiles) != len(outputFiles) {
		panic("Error: number of input and output files do not match!")
	}

	tests := make([]BootstrapGrangerMatrixTest, len(inputFiles))
	for i, inputFile := range inputFiles {
		tests[i] = ReadBootstrapGrangerMatrixInput(directory + "input/" + inputFile.Name())
	}

	for i, outputFile := range outputFiles {
		tests[i].ExpResults = ReadBootstrapGrangerMatrixOutput(directory + "output/" + outputFile.Name())
	}

	return tests
}

func ReadBootstrapGrangerMatrixInput(file string) BootstrapGrangerMatrixTest {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	K, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	T, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	lags, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	detTypeInt, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	nRep, _ := strconv.Atoi(line)

	line = skipComments(scanner)
	alpha, _ := strconv.ParseFloat(line, 64)

	line = skipComments(scanner)
	seed, _ := strconv.ParseInt(line, 10, 64)

	// Read Y data
	data := make([]float64, T*K)
	idx := 0
	for row := 0; row < T; row++ {
		line = skipComments(scanner)
		parts := strings.Fields(line)
		for _, p := range parts {
			data[idx], _ = strconv.ParseFloat(p, 64)
			idx++
		}
	}

	return BootstrapGrangerMatrixTest{
		K:             K,
		T:             T,
		Lags:          lags,
		DetType:       Deterministic(detTypeInt),
		NReplications: nRep,
		Alpha:         alpha,
		Seed:          seed,
		Y:             mat.NewDense(T, K, data),
	}
}

func ReadBootstrapGrangerMatrixOutput(file string) []struct{ CauseIdx, EffectIdx int } {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	line := skipComments(scanner)
	parts := strings.Fields(line)
	numResults, _ := strconv.Atoi(parts[1])

	results := make([]struct{ CauseIdx, EffectIdx int }, numResults)
	for i := 0; i < numResults; i++ {
		line = skipComments(scanner)
		parts = strings.Fields(line)
		results[i].CauseIdx, _ = strconv.Atoi(parts[0])
		results[i].EffectIdx, _ = strconv.Atoi(parts[1])
	}

	return results
}

func TestBootstrapGrangerMatrix(t *testing.T) {
	tests := ReadBootstrapGrangerMatrixTests("Tests/BootstrapGrangerMatrix/")
	for i, test := range tests {
		varNames := make([]string, test.K)
		for j := 0; j < test.K; j++ {
			varNames[j] = fmt.Sprintf("var%d", j)
		}

		ts := &TimeSeries{
			Y:        test.Y,
			VarNames: varNames,
		}

		spec := ModelSpec{
			Lags:          test.Lags,
			Deterministic: test.DetType,
		}

		rf, err := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
		if err != nil {
			t.Errorf("Test %d: Estimate failed: %v", i+1, err)
			continue
		}

		opts := GrangerBootstrapOptions{
			NReplications: test.NReplications,
			Alpha:         test.Alpha,
			Seed:          test.Seed,
		}

		results, err := rf.BootstrapGrangerMatrix(ts, opts)
		if err != nil {
			t.Errorf("Test %d: BootstrapGrangerMatrix returned error: %v", i+1, err)
			continue
		}

		if len(results) != test.K {
			t.Errorf("Test %d: len(results) = %d, want %d", i+1, len(results), test.K)
			continue
		}

		// Check diagonal is nil
		for j := 0; j < test.K; j++ {
			if results[j][j] != nil {
				t.Errorf("Test %d: results[%d][%d] should be nil (diagonal)", i+1, j, j)
			}
		}

		// Verify expected results
		for _, exp := range test.ExpResults {
			res := results[exp.CauseIdx][exp.EffectIdx]
			if res == nil {
				t.Errorf("Test %d: results[%d][%d] should not be nil", i+1, exp.CauseIdx, exp.EffectIdx)
				continue
			}
			if res.BootPValue < 0 || res.BootPValue > 1 {
				t.Errorf("Test %d: BootPValue out of range: %v", i+1, res.BootPValue)
			}
		}
	}
}
