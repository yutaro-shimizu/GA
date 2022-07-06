from 01GA import main

generations = [10,100,500,1000]
population = [10,100,500,1000]
layers = [1, 5, 10, 50]
percent = [0.1, 0.2, 0.3, 0.5]
mutation = [0.01, 0.05, 0.1, 0.5]
increment = [0.001, 0.01, 0.05, 0.1]

if __name__ == "__main__":
    for gen in generations:
        for ind in population:
            for layer in layers:
                for per in percent:
                    for mut in mutation:
                        for inc in increment:
                            main(gen, ind, ...)