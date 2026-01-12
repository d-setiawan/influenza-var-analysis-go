# Authors: Rohan Adla, Arrio Gonsalves, Shreyan Nalwad, Dylan Setiawan
# Date: Dec 12th 2025
# Project: A VAR-based Computational Analysis of Influenza and Weather Dynamics
# Class: 02-613 at Caregie Mellon University

# Check if packages are installed; if not, install themß
if (!require("shiny")) {
  install.packages("shiny")
}

if (!require("ggplot2")) {
  install.packages("ggplot2")
}

if (!require("wesanderson")) {
  install.packages("wesanderson")
}

if (!require("reshape2")) {
  install.packages("reshape2")
}

if (!require("tidyverse")) {
  install.packages("tidyverse")
}

if (!require("gganimate")) {
  install.packages("gganimate")
}

if (!require("gifski")) {
  install.packages("gifski")
}

library(shiny)
library(ggplot2)
library(wesanderson)
library(ggplot2)
library(gganimate)
library(tidyverse)
library(reshape2)
library(gifski)

# Setup UI 
ui <- fluidPage(
  titlePanel("Influenza Autoregression Analysis"),
    
  sidebarLayout(
    # Dropdown to select country
    sidebarPanel(
      selectInput(
        inputId = "country", # Unique ID for the input
        label = "Select a Country for the analysis:", # Label displayed to the user
        choices = c("Singapore", "Qatar")
      ),
      selectInput(
        inputId = "influenza", # Unique ID for the input
        label = "Select an Influenza Strain for analysis:", # Label displayed to the user
        choices = c("Influenza A", "Influenza B") # List of available choices
      ),
      actionButton("runGoCode", "Run Autoregression")
    ),
    
    mainPanel(
      plotOutput("outputPlot"),
      plotOutput("grangerPlot"),
      imageOutput("irfPlot"),
    )
  )
)
 
# Setup server logic
server <- function(input, output) {
  
  observeEvent(input$runGoCode, {
    # Determine the country selected by the user
    country <- input$country
    # If influenza A, use log_diff_a, else log_diff_b
    influenza_type <- ifelse(input$influenza == "Influenza A", "A", "B")
    influenza_column <- ifelse(input$influenza == "Influenza A", "LOG_INF_A", "LOG_INF_B")
    
    # Compile the Go program
    compile_result <- system("go build", intern = TRUE)
    print("Go program compiled successfully.")
    
    # Run the compiled Go program
    print("Running Go program...")
    run_result <- system(paste("./application ", as.character(country), " ", as.character(influenza_type), sep = ""),
      intern = TRUE)
    print("Go program executed successfully.")
    
    # Read the CSV files
    old_data <- read.csv(paste0("../Files/Final_Training_Data/", country, "/",
      "Training_Data_INF_", influenza_type, "_transformed.csv"))
    forcast_data <- read.csv("../Files/Output/forecast_results.csv")
    granger_data <- read.csv("../Files/Output/granger_results.csv")
    irf_data <- read.csv("../Files/Output/irf_results.csv")

    # see if data was loaded correctly
    if (nrow(old_data) == 0 || nrow(forcast_data) == 0 || nrow(granger_data) == 0 || nrow(irf_data) == 0) {
      stop("Error: One or more data files are empty.")
    }
    print("Data loaded successfully.")

    # Get most recent 30 days of old data
    recent_old_data <- tail(old_data, 30)

    # Add a Week variable if not present
    forcast_data$Week <- seq_len(nrow(forcast_data))
    recent_old_data$Week <- seq_len(nrow(recent_old_data))

    # Reshape data for plotting
    melted_forcast <- melt(forcast_data, id.vars = "Week", measure.vars =
      c(influenza_column))
    melted_recent <- melt(recent_old_data, id.vars = "Week", measure.vars =
      c(influenza_column))

    plot <- ggplot() +
        geom_line(data = melted_recent, aes(x = Week, y = value, color = variable),
            size = 1) +
        geom_line(data = melted_forcast, aes(x = Week + nrow(recent_old_data), y =
            value, color = variable),
            linetype = "dashed", size = 1) +
        scale_color_manual(values = wes_palette("Darjeeling1", n = 2))
        labs(title = paste("Influenza Forecasting in", country),
            x = "Weeks",
            y = "Log-Differenced Influenza Cases",
            color = "Legend") +
        theme_minimal()


    # Animated IRF line graph showing impact over horizons for each variable's shock
    irf_data_df_long <- irf_data %>%
    pivot_longer(
      cols = -Horizon,
      names_to = "Shock",
      values_to = "Impact"
    )

    plot_irf <- ggplot(irf_data_df_long, aes(x = Shock, y = Impact, fill = Shock)) +
    geom_col() +
    labs(title = "Impulse Response Function (Bar Animation)",
         subtitle = "Horizon: {closest_state}",
         x = "Shock Variable",
         y = "Impact on Influenza") +
    transition_states(Horizon, transition_length = 2, state_length = 1) + # animate by Horizon
    ease_aes('cubic-in-out') +
    theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust = 1))

  output$irfPlot <- renderImage({
    # animate on the fly and return as image
    outfile <- tempfile(tmpdir = "../Files/Images", fileext = ".gif")
    animate(plot_irf,
            renderer = gifski_renderer(outfile),
            fps = 10,       # slower for bars
            width = 900,
            height = 600)
    list(src = outfile,
         contentType = "image/gif",
         width = 900,
         height = 600)
  }, deleteFile = TRUE)

    # Cumulative IRM bar graph that shows total impact all horizons for 
    # each variable and its absolute impact on influenza variable
    irmcum_data <- irf_data %>%
      select(-Horizon) %>%
      summarise(across(everything(), sum)) %>%
      pivot_longer(
        cols = everything(),
        names_to = "Variable",
        values_to = "CumulativeImpact"
      )

    # this plot is not rendered in this version of the app, but is included for future use incase you 
    # wanna see the cumulative impact magnitudes
    plot_irmcum <- ggplot(irmcum_data, aes(x = Variable, y = CumulativeImpact, fill = Variable)) +
      geom_bar(stat = "identity") +
      labs(title = paste("Cumulative Impulse Response Magnitudes in", country),
           x = "Variable",
           y = "Cumulative Impact") +
      theme_minimal()

    # granger causality results as a heatmap
    plot_granger <- ggplot(granger_data, aes(x = CauseVar, y = EffectVar, fill = PValue)) +
      geom_tile() +
      geom_text(aes(label = ifelse(Significant, "GC", "")), color = "white") +
      scale_fill_gradient(low = "red", high = "yellow") +
      labs(title = paste("Granger Causality Results in", country),
           x = "CauseVar",
           y = "EffectVar",
           fill = "PValue") +
      theme_minimal() + 
      theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust = 1))

    

    # Render the plot in the Shiny app
    output$outputPlot <- renderPlot({
      plot
    })
    output$grangerPlot <- renderPlot({
      plot_granger
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
