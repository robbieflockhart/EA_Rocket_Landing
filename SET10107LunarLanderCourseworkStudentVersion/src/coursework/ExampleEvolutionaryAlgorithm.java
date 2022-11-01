package coursework;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;
import java.util.stream.*;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * You Can Use This Class to implement your EA or implement your own class that extends {@link NeuralNetwork} 
 * 
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {
	

	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {		
		//Initialise a population of Individuals with random weights
		population = initialise();

		//Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */		
		
		while (evaluations < Parameters.maxEvaluations) {

			/**
			 * this is a skeleton EA - you need to add the methods.
			 * You can also change the EA if you want 
			 * You must set the best Individual at the end of a run
			 * 
			 */

			// Select 2 Individuals from the current population. Currently returns random Individual
			//Individual parent1 = select(); 
			//Individual parent2 = select();
			
			Individual parent1 = tournamentSelection(Parameters.tournamentSize); 
			Individual parent2 = tournamentSelection(Parameters.tournamentSize);
			
			//Individual parent1 = rouletteSelection();
			//Individual parent2 = rouletteSelection();
			
			// Generate a child by crossover. Not Implemented			
			//ArrayList<Individual> children = reproduce(parent1, parent2);	
			
			//ArrayList<Individual> children = onePointXOver(parent1, parent2);
			
			//ArrayList<Individual> children = twoPointXOver(parent1, parent2);
			
			ArrayList<Individual> children = uniformXOver(parent1, parent2);
			
			//mutate the offspring
			//mutate(children);
			
			swapMutation(children);
			
			//inversionMutation(children);
			
			// Evaluate the children
			evaluateIndividuals(children);			

			// Replace children in population
			//replace(children);
			
			replaceIfBetter(children);

			// check to see if the best has improved
			best = getBest();
			
			// Implemented in NN class. 
			outputStats();
			
			//Increment number of completed generations			
		}

		//save the trained network to disk
		saveNeuralNetwork();
	}

	

	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}


	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest() {
		best = null;;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}

	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}

	/**
	 * Selection --
	 * 
	 * NEEDS REPLACED with proper selection this just returns a copy of a random
	 * member of the population
	 */
	private Individual select() {		
		Individual parent = population.get(Parameters.random.nextInt(Parameters.popSize));
		return parent.copy();
	}
	//Tournament Selection - any tournament size 
	private Individual tournamentSelection(int tournamentSize) {
		ArrayList<Individual> tournament = new ArrayList<>();
		for (int i = 0; i < tournamentSize; i++) {
			Individual potentialParent = population.get(Parameters.random.nextInt(Parameters.popSize));
			tournament.add(potentialParent.copy());
		}
		Individual parent = null;
		double bestFitness = 100.0;
		for (int i = 0; i < tournament.size(); i ++) {
			if (tournament.get(i).fitness < bestFitness) {
				bestFitness = tournament.get(i).fitness;
				parent = tournament.get(i).copy();
			}
		}
		return parent.copy();
	}
	//Roulette Wheel Selection
	private Individual rouletteSelection() {
		Individual parent = null;
		double fitnessSum = 0.0;
		for (Individual individual : population) {
			fitnessSum += individual.fitness;
		}
		double pointer = fitnessSum * Parameters.random.nextDouble();
		double partialFitnessSum = 0.0;
		for (Individual individual : population) {
			partialFitnessSum += individual.fitness;
			if (partialFitnessSum >= pointer) {
				parent = individual;
				break;
			}
		}
		return parent.copy();
	}

	/**
	 * Crossover / Reproduction
	 * 
	 * NEEDS REPLACED with proper method this code just returns exact copies of the
	 * parents. 
	 */
	private ArrayList<Individual> reproduce(Individual parent1, Individual parent2) {
		ArrayList<Individual> children = new ArrayList<>();
		children.add(parent1.copy());
		children.add(parent2.copy());			
		return children;
	} 
	//One Point Crossover
	private ArrayList<Individual> onePointXOver(Individual parent1, Individual parent2) {
		ArrayList<Individual> children = new ArrayList<>();
		Individual child1 = parent1.copy();
		Individual child2 = parent2.copy();
		if (Parameters.random.nextDouble() <= Parameters.xOverRate) {
			int xOverPoint = Parameters.random.nextInt(parent1.chromosome.length);
			for (int i = 0; i < parent1.chromosome.length; i++) {
				if (i < xOverPoint) {
					child1.chromosome[i] = parent1.chromosome[i];
					child2.chromosome[i] = parent2.chromosome[i];
				}
				else {
					child1.chromosome[i] = parent2.chromosome[i];
					child2.chromosome[i] = parent1.chromosome[i];
				}
			}
		}
		children.add(child1.copy());
		children.add(child2.copy());			
		return children;
	} 
	//Two Point Crossover
	private ArrayList<Individual> twoPointXOver(Individual parent1, Individual parent2) {
		ArrayList<Individual> children = new ArrayList<>();
		Individual child1 = parent1.copy();
		Individual child2 = parent2.copy();
		if (Parameters.random.nextDouble() <= Parameters.xOverRate) {
			int xOverPoint1 = Parameters.random.nextInt(parent1.chromosome.length - 1);
			int xOverPoint2 = Parameters.random.nextInt((parent1.chromosome.length - xOverPoint1) + xOverPoint1);
			for (int i = 0; i < parent1.chromosome.length; i++) {
				if (i < xOverPoint1 || i > xOverPoint2) {
					child1.chromosome[i] = parent1.chromosome[i];
					child2.chromosome[i] = parent2.chromosome[i];
				}
				else {
					child1.chromosome[i] = parent2.chromosome[i];
					child2.chromosome[i] = parent1.chromosome[i];
				}
			}
		}
		children.add(child1.copy());
		children.add(child2.copy());			
		return children;
	} 
	//Uniform Crossover
	private ArrayList<Individual> uniformXOver(Individual parent1, Individual parent2) {
		ArrayList<Individual> children = new ArrayList<>();
		Individual child1 = parent1.copy();
		Individual child2 = parent2.copy();
		if (Parameters.random.nextDouble() <= Parameters.xOverRate) {
			for (int i = 0; i < parent1.chromosome.length; i++) {
				double probability = Parameters.random.nextDouble();
				if (probability <= 0.5) {
					child1.chromosome[i] = parent1.chromosome[i];
					child2.chromosome[i] = parent2.chromosome[i];
				}
				else {
					child1.chromosome[i] = parent2.chromosome[i];
					child2.chromosome[i] = parent1.chromosome[i];
				}
			}
		}
		children.add(child1.copy());
		children.add(child2.copy());			
		return children;
	} 
	
	/**
	 * Mutation
	 * 
	 * 
	 */
	private void mutate(ArrayList<Individual> individuals) {		
		for(Individual individual : individuals) {
			for (int i = 0; i < individual.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					if (Parameters.random.nextBoolean()) {
						individual.chromosome[i] += (Parameters.mutateChange);
					} else {
						individual.chromosome[i] -= (Parameters.mutateChange);
					}
				}
			}
		}		
	}	
	//Swap Mutation
	private void swapMutation(ArrayList<Individual> individuals) {		
		for(Individual individual : individuals) {
			for (int i = 0; i < individual.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					int swapPoint1 = i;
					int swapPoint2 = Parameters.random.nextInt(individual.chromosome.length);
					double tempGene = 0.0;						
					tempGene = individual.chromosome[swapPoint1];
					individual.chromosome[swapPoint1] = individual.chromosome[swapPoint2];
					individual.chromosome[swapPoint2] = tempGene;
				}
			}
		}
	}	
	//Inversion Mutation
	private void inversionMutation(ArrayList<Individual> individuals) {		
		for(Individual individual : individuals) {
			if (Parameters.random.nextDouble() < Parameters.mutateRate) {
				int inversionPoint1 = Parameters.random.nextInt(individual.chromosome.length - 1);
				int inversionPoint2 = Parameters.random.nextInt((individual.chromosome.length - inversionPoint1) + inversionPoint1);
				double[] tempArray1 = new double[inversionPoint2 - inversionPoint1];
				int j = inversionPoint1;
				for (int i = 0; i < tempArray1.length; i++) {
					tempArray1[i] = individual.chromosome[j];
					j += 1;
				}
				double[] tempArray2 = new double[inversionPoint2 - inversionPoint1];
				int k = tempArray1.length - 1;
				for (int i = 0; i < tempArray1.length; i++) {
					tempArray2[i] = tempArray1[k];
					k -= 1;
				}
				int l = inversionPoint1;
				for (int i = 0; i < tempArray2.length; i++) {
					individual.chromosome[l] = tempArray2[i];
					l += 1;
				}
			}
		}
	}
	
	
	/**
	 * 
	 * Replaces the worst member of the population 
	 * (regardless of fitness)
	 * 
	 */
	private void replace(ArrayList<Individual> individuals) {
		for(Individual individual : individuals) {
			int idx = getWorstIndex();		
			population.set(idx, individual);
		}		
	}
	//Replace worst in population only if child is better
	private void replaceIfBetter(ArrayList<Individual> individuals) {
		for(Individual individual : individuals) {
			int idx = getWorstIndex();	
			if (individual.fitness <= population.get(idx).fitness) {
				population.set(idx, individual);
			}
		}		
	}

	

	/**
	 * Returns the index of the worst member of the population
	 * @return
	 */
	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i; 
			}
		}
		return idx;
	}	

	@Override
	public double activationFunction(double x) {
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		return Math.tanh(x);
	}
}
