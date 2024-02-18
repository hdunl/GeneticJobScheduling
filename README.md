# Genetic Algorithm for Job Scheduling

This repository contains a Genetic Algorithm (GA) implementation in Python for solving job scheduling problems. The GA aims to optimize the assignment of job operations to machines while minimizing the makespan, i.e., the total time taken to complete all jobs.

## Overview

The GA is designed to handle job scheduling problems where each job consists of multiple operations, each requiring specific machines and durations. The goal is to find an optimal assignment of operations to machines such that the total completion time, or makespan, is minimized.

## Features

- **Initialization**: Random initialization of the population with diverse solutions.
- **Selection**: Fitness-proportionate selection mechanism for parent selection.
- **Crossover**: Single-point crossover for generating offspring solutions.
- **Mutation**: Random mutation of individuals to introduce diversity in the population.
- **Elitism**: Preserving the best solution from the previous generation.
- **Fitness Calculation**: Evaluation of the fitness of each individual based on the makespan.
- **Convergence Criteria**: Termination based on a maximum number of generations or no improvement threshold.
- **Visualization**: Plotting the fitness history and solution distribution.

## Usage

1. Clone the repository

2. Install the required dependencies:

3. Configure the job scheduling problem by modifying the `jobs` variable in the `main.py` file.

4. Run the GA:

5. Analyze the results, including the best solution, makespan, fitness history, and solution distribution.

## Example

An example job scheduling problem is provided in the `main.py` file. You can adjust the job parameters, population size, mutation rate, and convergence criteria according to your specific requirements.


