import numpy as np
import operator
import math
import random
import scipy as sp
from random import shuffle
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from scipy.misc import imread, imsave
from scipy.ndimage.filters import sobel, laplace, gaussian_laplace


class GP:

    def __init__(self, num_gens, pop_size, cross_rate, mut_rate):
        self.cx_rate = cross_rate
        self.m_rate = mut_rate
        self.pop = pop_size
        self.gens = num_gens

    def setData(self, all_data, labels):
        self.data = all_data
        self.data_labels = labels

    def setTrain(self, train_points, ground_truth_img):
        self.points = train_points
        self.gt_img = ground_truth_img

    # define the evaluation function
    # counts the number of correct classifications
    # Method must return an array
    def evalFunc(self, individual, all_data, train_points):
        func = self.toolbox.compile(expr=individual)
        total = 1.0

        for point in train_points:
            #print("Point", point[0], point[1])
            tmp = [float(x) for x in all_data[point[1]][point[0]][1:]]
            if (int((0 if (func(*tmp)) <= 0 else 1) == point[2])):
                if (point[2] == 0):
                    total += 0.1
                else:
                    total += 0.9
                #print("Result: " + str(int((0 if (func(*tmp)) <= 0 else 1)) + "\tAns: " + str(point[2]))


        return total,

    # define the evaluation for the testing
    # Identical to above, but does more logging
    def testEval(self, individual, all_data, train_points, test):
        tp, tn, fp, fn = 0,0,0,0
        func = self.toolbox.compile(expr=individual)
        total = 0
        im = []

        for y in range(len(self.gt_img)):
            #print("In row: " + str(y))
            t = []
            for x in range(len(self.gt_img[y])):
                tmp = [float(a) for a in all_data[y][x][1:]]
                res = 0 if (func(*tmp)) <= 0 else 1
                if (res == 1):
                    if (all_data[y][x][0] > 0):
                        tp += 1
                        t.append([125,0,0,255])
                    else:
                        fp += 1
                        t.append([0,125,0,255])
                else:
                    if (all_data[y][x][0] > 0):
                        fn += 1
                        t.append([0,0,125,255])
                    else:
                        tn += 1
                        t.append([0,0,0,255])
            im.append(t)
            #total += res

        for i in train_points:
            im[i[1]][i[0]] = (255, 255, 0, 255)
        imsave("Results/"+self.test_name + "-" + str(self.test_num) + "-res_" + str(test) + ".png", im)
        print("Image saved")
        return [(tp+tn)/(tp+tn+fp+fn), tp, tn, fp, fn]


    # Definition of the protected div
    def protectedDiv(self, left, right):
        # if the number is close to 0, treat is as 0 to prevent infinity
        try:
            if (right < 0.000000000001):
                return 1
            return left / right
        except ZeroDivisionError:
            return 1
        except RuntimeWarning:
            print("Left: " + str(left) + "\tRight: " + str(right))
            return 1

    # defines the absolute square root
    def abs_sqrt(self, num):
        return math.sqrt(abs(num))

    def cbrt(self, num):
        return sp.special.cbrt(num)

    # defines sin using radians
    def sin(self, num):
        try:
            n2 = math.radians(num)
            return math.sin(n2)
        except ValueError:
            print("Infinity Warning")
            print("Num: " + str(num))

    # defines cosine using radians
    def cos(num):
        try:
            n2 = math.radians(num)
            return math.cos(n2)
        except ValueError:
            print("Infinity Warning")
            print("Num: " + str(num))

    # Defines the modulo operator
    def modulo(self, n1, n2):
        if (n2 == 0):
            return n1
        return n1%n2

    # Defines distance calculation using 4 inputs
    def dist(x1,y1,x2,y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    # Defines an if statement comparing to 0
    def ifStatement(self, n1, n2, n3):
        if (n1 < 0):
            return n2
        else:
            return n3

    # defines an if statement comparing 2 numbers
    def compStatement(n1, n2, n3, n4):
        if (n1 < n2):
            return n3
        else:
            return n4

    # calculates the determinant of a 2x2 matrix made from 4 inputs
    def det2x2(n11, n12, n21, n22):
        return (n11*n22-n12*n21)

    def average(n1, n2):
        return (n1+n2)/2

    def toDict(self):
        d = {}
        for i in range(len(self.data_labels)):
            d["ARG"+str(i)] = self.data_labels[i]
        return d






    def setUpGP(self, test_name, test_num):

        self.test_name = test_name
        self.test_num = test_num
        # allows to set the test number to use, and the base output name
        #test_num = 6;
        name = str(test_name)+"Experiment-" + str(test_num)


        # set up GP parameters
        # Original Value, Original Mean x 3, Value Sobel vertical, value sobel horizontal, edge detect, blotchy value
        pset = gp.PrimitiveSet("MAIN", len(self.data_labels)-1)
        # add the addition operator. Has 2 inputs
        # sets up the rest of the operators to use
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(self.protectedDiv, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(operator.neg, 1)

        # include other operators based on test number
        if (test_num > 1):
            pset.addPrimitive(self.sin,1)
        if (test_num > 3):
            pset.addPrimitive(self.cbrt, 1)
        if (test_num > 4):
            pset.addPrimitive(round, 1)
        if (test_num > 5):
            pset.addPrimitive(self.ifStatement,3)
        if (test_num > 2):
            pset.addPrimitive(math.tanh,1)
        #if (test_num > 6):
        #    pset.addPrimitive(self.modulo, 2)
        # unused operators
        #pset.addPrimitive(cos,1)
        #pset.addPrimitive(abs_sqrt, 1)
        #pset.addPrimitive(max, 2)
        #pset.addPrimitive(min, 2)
        #pset.addPrimitive(math.floor, 1)
        #pset.addPrimitive(math.ceil, 1)
        #pset.addPrimitive(modulo, 2) # seems to be pretty good
        #pset.addPrimitive(dist,4)
        #pset.addPrimitive(compStatement,4)
        #pset.addPrimitive(det2x2,4)
        #pset.addPrimitive(average,2)

        # add the posibility of a constant from -1 to 1
        pset.addEphemeralConstant("const", lambda: random.uniform(-1, 1))

        #pset.renameArguments(self.data_labels)
        pset.renameArguments(**(self.toDict()))#, "mean17_17", "mean21_21", "std13_13", "std17_17", "std21_21","sobel_v","sobel_h","laplace","gaussian_laplace"])


        # state that it is a maximization problem
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # set the individuals language and fitness type
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        # set tree generation parameters
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
        # Set up individuals and populations and set up the compilation process
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        self.toolbox = toolbox
        # register the evaluation function
        toolbox.register("evaluate", self.evalFunc, all_data=self.data, train_points=self.points)

        # set up GP parameters
        toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        # Set up logging
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        # register methods for calculating various statistics
        mstats.register("mean", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        self.mstats = mstats

    def runGP(self):
        print ("Running GP")
        # open file summary
        f_avg = open('Logs/'+self.test_name+'-'+str(self.test_num)+'-avg.txt', 'w+')
        f_individuals = open('Logs/'+self.test_name+'-'+str(self.test_num)+'-individuals.txt', 'w+')
        avg_vals = []


        # run the test 20 times
        for i in range(20):
            # split the data each time and register the evaluation function using the correct data
            #folds, test_data = splitData(all_data, fold_k)
            #toolbox.register("evaluate", evalFunc, data)

            # generate the populations
            pop = self.toolbox.population(n=self.pop)
            # holds the n best individuals
            hof = tools.HallOfFame(1)
            print ("Run: " + str(i))
            pop, logs = algorithms.eaSimple(pop, self.toolbox, self.cx_rate, self.m_rate, self.gens, stats=self.mstats, halloffame=hof, verbose=True)
            # Open file to log the results
            f = open('Logs/'+self.test_name+'-'+str(self.test_num)+'-logs-' + str(i) +'.txt', 'w')
            f.write(str(logs))
            f.close()
            expr = hof[0]
            # Print and store the testing results for the best solution
            f_individuals.write(str(expr)+"\n")
            #print("fitness: " + str(testEval(expr, test_data)))
            avg_vals.append("\t".join(str(s) for s in self.testEval(expr, self.data, self.points, i)))

        # write results to the file
        f_avg.write(str("\n".join(str(s) for s in avg_vals)))
        f_avg.close()
