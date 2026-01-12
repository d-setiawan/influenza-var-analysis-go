# Influenza VAR Causality Analysis (Go + R Shiny)

This project is a small toolkit for running **Vector Autoregression (VAR)**-based analyses on influenza and weather data in Go, with an R Shiny dashboard for visualization.  

The Go side loads preprocessed CSVs, fits a VAR model, produces forecasts, impulse response functions (IRFs), Granger causality tests, and bootstrap confidence intervals, and writes everything to CSV. The R Shiny app compiles/runs the Go program and turns those CSVs into interactive plots. 

## Attribution

This repository is a curated, portfolio-ready version of a group project
developed in collaboration with:

- Rohan Adla
- Arrio Gonsalves
- Shreyan Nalwad

Original repository:
https://github.com/ADGArrio/Influenza_Causality_AR_Project

My contributions include:
- VAR model estimation in Go
- Impulse response functions (IRFs)
- Residual bootstrapping with parallel workers
- Granger causality testing and bootstrap inference
- Weather Data Retrieval

## Project Demonstration

Watch a full walkthrough of the project in action:

[![Project Demo](https://img.shields.io/badge/Demo-Google%20Drive-blue?style=for-the-badge&logo=googledrive)](https://drive.google.com/file/d/1ymkJm-H81Vskn2aj7yfR8IJK_v2HPxSj/view?usp=sharing)

--- 

## Repository layout (key pieces)

From the point of view of this project, the important files are:

- `application/`
  - `main.go` — command-line driver: loads data, fits VAR, runs forecasts, IRFs, Granger tests, bootstraps, and writes CSV outputs. :contentReference[oaicite:1]{index=1}
  - `datatypes.go` — core data structures and interfaces (`TimeSeries`, `ModelSpec`, `ReducedFormVAR`, `ReducedForm`, `Estimator`, bootstrap structs, etc.). :contentReference[oaicite:2]{index=2}
  - `functions.go` — main methods for forecasting, IRFs, VAR estimation, Granger causality, and bootstrap logic. :contentReference[oaicite:3]{index=3}
  - `io.go` — CSV loading and CSV writers (forecasts, IRFs, Granger matrices, bootstrap results) plus text summary/printing helpers. :contentReference[oaicite:4]{index=4}
  - (compiled binary is named `application` by default)
- `app.R` — R Shiny app that compiles/runs the Go binary and visualizes forecast, Granger, and IRF outputs. :contentReference[oaicite:5]{index=5}
- `../Files/Final_Training_Data/` — input CSVs (per-country, per-strain; paths are hard-coded in `main.go`).
- `../Files/Output/` — where all Go-generated CSV outputs are written.

---

## Prerequisites

### Go

- Go (1.20+ recommended).
- Gonum libraries (used for matrices and distributions):  
  - `gonum.org/v1/gonum/mat`  
  - `gonum.org/v1/gonum/stat/distuv`   

Install (from inside `application/`, if you don’t already have a `go.mod`):

```bash
go mod init influenza-var-analysis   # or any module path you like
go get gonum.org/v1/gonum@latest
go mod tidy
```

### R
```R
install.packages(c("shiny", "ggplot2", "wesanderson",
                   "reshape2", "tidyverse",
                   "gganimate", "gifski"))
```


# Quickstart Guide

### 1. Install prerequisites

### 2. Build Go Program
```bash
go build -o application
```

### 3. Run VAR analysis from command line
```bash
./application Singapore A
```
This runs an analysis on the pre-existing dataset for Singapore and influenza type A.  
All outputs will appear in:
```
Files/Output/
```

### 4. Run the Shiny Dashboard
This app is separate from the command line interface so you can run it and it will immediately pull up any dataset that the app analyzes with options of graphs.
```R
setwd("application")
source("app.R")
```

### 5. Adding your own CSV files
You can put CSV files in this folder:  

``
Files/Final_Training_Data/<NewCountry>/Training_Data_INF_A_transformed.csv
Files/Final_Training_Data/<NewCountry>/Training_Data_INF_B_transformed.csv
``  

After which you can register the new country inside `main.go` and add the country name to the dropdown in `app.R`

1) Create a folder (let's use India):
`mkdir -p Files/Final_Training_Data/India`
2) Add CSV's:
```
Files/Final_Training_Data/India/Training_Data_INF_A_transformed.csv
Files/Final_Training_Data/India/Training_Data_INF_B_transformed.csv
```
3) Modify main.go
Find this block:
```Go
switch country {
case "Singapore":
    filename = "Singapore/Training_Data_INF_"
case "Qatar":
    filename = "Qatar/Training_Data_INF_"
default:
    panic("Unsupported country")
}
```
4) Add
```Go
case "India":
    filename = "India/Training_Data_INF_"
```
5) Modify app.R
```R
choices = c("Singapore", "Qatar", "India")
```

## Repo Layout
```
application/
	main.go               	# CLI driver: loads data, runs VAR, writes outputs
	datatypes.go         		# Core structs and interfaces
	functions.go         		# VAR estimation, forecasting, IRF, Granger, bootstrapping
	io.go                		# CSV loader, writers, summary printing
	app.R                		# Shiny app for visualization
    functions_test.go           # Tests for all core Go functions
    Tests                       # Input and Output files for all function_tests

Files/
	Final_Training_Data/ 		# Input CSVs (processed influenza + weather data)
	Output/             		# Generated output CSVs (forecasts, IRFs, GC, bootstraps)
    Raw Data/               # Lightly processed data from WHO and NOAA 

Data_Processing/
	Assumptions_Checking/		
		assumptions_checking.py	# Checks for VAR assumptions and data validity
	Data_Joining/       		# Script merging influenza + weather datasets
	Data_Cleanup/       		# General preprocessing
	WeatherData_cleanup/		# NOAA-specific cleanup and normalization scripts

```

## Summary of Repository Layout
- **application/**  
  All Go and R source code for VAR modeling, forecasting, IRFs, Granger causality,
  and bootstrap-based uncertainty quantification.

- **Data_Processing/**  
  Scripts used to clean, transform, and validate influenza and weather datasets
  before feeding them into the VAR pipeline.

- **Files/Final_Training_Data/**  
  The canonical training datasets used for model estimation.

- **Files/Output/**  
  Automatically generated results from running the Go application.

- **Files/Raw Data/**  
  Lightly processed input files retrieved from surveillance databases and NOAA.