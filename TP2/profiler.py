import line_profiler, ejercicio1


profiler = line_profiler.LineProfiler()


profiler.add_function(ejercicio1.set_probabilities)
profiler.add_function(ejercicio1.entropy)
profiler.add_function(ejercicio1.gain)
profiler.add_function(ejercicio1.calculate_gain_and_entropy)
profiler.add_function(ejercicio1.get_max_gain)
profiler.add_function(ejercicio1.id3)
wrapper = profiler(ejercicio1.main)
wrapper()


profiler.print_stats(open("line_profiler","w"), output_unit=1)