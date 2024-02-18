import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def initialize_population(pop_size, jobs):
    population = []
    for _ in range(pop_size):
        individual = []
        for job_id, job in enumerate(jobs):
            for operation_id in job['operations']:
                machine_options = job['machine_options'][operation_id]
                machine = np.random.choice(machine_options)
                individual.append((job_id, operation_id, machine))
        np.random.shuffle(individual)
        population.append(individual)
    return population


def plot_solution_distribution(population):
    solutions = [tuple(ind) for ind in population]
    solution_counts = Counter(solutions)
    plt.bar(range(len(solution_counts)), solution_counts.values())
    plt.title('Solution Distribution')
    plt.xlabel('Solution')
    plt.ylabel('Frequency')
    plt.xticks(range(len(solution_counts)), solution_counts.keys(), rotation=45)
    plt.show()


def calculate_makespan(individual, jobs):
    machine_availability = [0] * 10
    job_completion = [0] * len(jobs)
    for job_id, operation_id, machine in individual:
        start_time = max(machine_availability[machine], job_completion[job_id])
        duration = jobs[job_id]['durations'][operation_id]
        end_time = start_time + duration
        machine_availability[machine] = end_time
        job_completion[job_id] = end_time
    makespan = max(machine_availability)
    return makespan


def calculate_fitness(individual, jobs):
    makespan = calculate_makespan(individual, jobs)
    max_fitness = 1 / min(jobs[j]['durations'][op] for j in range(len(jobs)) for op in jobs[j]['operations'])
    return 1 / makespan / max_fitness


def selection(population, fitnesses, num_parents):
    selection_probs = fitnesses / np.sum(fitnesses)
    parent_indices = np.random.choice(len(population), size=num_parents, replace=False, p=selection_probs)
    parents = [population[index] for index in parent_indices]
    return parents


def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, min(len(parent1), len(parent2)))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual, mutation_rate, jobs):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            job_id, operation_id, _ = individual[i]
            new_machine = np.random.choice(jobs[job_id]['machine_options'][operation_id])
            individual[i] = (job_id, operation_id, new_machine)
    np.random.shuffle(individual)
    return individual


def genetic_algorithm(jobs, pop_size=200, initial_mutation_rate=0.1, improvement_threshold=0.001,
                      max_no_improvement_generations=300):
    population = initialize_population(pop_size, jobs)
    best_fitness_history = []
    no_improvement_count = 0
    mutation_rate = initial_mutation_rate
    generation = 0
    last_best_fitness = 0

    while no_improvement_count < max_no_improvement_generations:
        fitnesses = np.array([calculate_fitness(ind, jobs) for ind in population])
        current_best_fitness = np.max(fitnesses)
        best_fitness_history.append(current_best_fitness)

        if current_best_fitness - last_best_fitness < improvement_threshold:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            last_best_fitness = current_best_fitness

        parents = selection(population, fitnesses, pop_size // 2)
        next_generation = [population[np.argmax(fitnesses)]]  # Elitism

        while len(next_generation) < pop_size:
            parent_indices = np.random.choice(range(len(parents)), size=2, replace=False)
            parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]
            child1, child2 = crossover(parent1, parent2)
            next_generation += [mutate(child1, mutation_rate, jobs), mutate(child2, mutation_rate, jobs)]

        population = next_generation[:pop_size]

        mutation_rate *= 0.98

        if generation % 10 == 0 or no_improvement_count >= max_no_improvement_generations:
            print(f"Generation {generation}: Best Fitness = {current_best_fitness:.4f}")

        generation += 1

    best_index = np.argmax([calculate_fitness(ind, jobs) for ind in population])
    best_solution = population[best_index]
    best_makespan = calculate_makespan(best_solution, jobs)

    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_history, marker='o', linestyle='-')
    plt.title('Fitness History')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.show()

    plot_solution_distribution(population)

    return best_solution, best_makespan, best_fitness_history


jobs = [
    {'operations': [0, 1, 2], 'durations': [3, 5, 4], 'machine_options': [[0, 1], [1, 2], [2, 0]]},
    {'operations': [0, 1, 2, 3], 'durations': [2, 4, 3, 6], 'machine_options': [[1], [0, 2], [1, 3], [3, 0]]},
    {'operations': [0, 1], 'durations': [4, 3], 'machine_options': [[0, 2], [1, 3]]},
    {'operations': [0, 1, 2], 'durations': [5, 3, 2], 'machine_options': [[0, 1], [1, 3], [2, 3]]},
    {'operations': [0, 1, 2, 3, 4], 'durations': [3, 4, 2, 5, 6], 'machine_options': [[1, 3], [0, 2], [1], [2, 3], [0]]},
    {'operations': [0, 1], 'durations': [2, 3], 'machine_options': [[0, 2], [1, 3]]},
    {'operations': [0, 1, 2, 3], 'durations': [4, 3, 5, 2], 'machine_options': [[1, 3], [0, 2], [1, 3], [0]]},
    {'operations': [0, 1, 2, 3], 'durations': [3, 2, 4, 5], 'machine_options': [[0, 1], [1, 3], [0, 2], [2, 3]]},
    {'operations': [0, 1], 'durations': [2, 4], 'machine_options': [[0, 2], [1, 3]]},
    {'operations': [0, 1, 2], 'durations': [3, 5, 4], 'machine_options': [[0, 1], [1, 2], [2, 3]]}
]

best_solution, best_makespan, fitness_history = genetic_algorithm(jobs)
print(f"Best Solution: {best_solution}")
print(f"Best Makespan: {best_makespan:.2f}")
