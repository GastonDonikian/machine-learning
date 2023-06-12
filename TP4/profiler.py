import line_profiler, preprocess
import algorithms.hierarchical_clustering as hc


profiler = line_profiler.LineProfiler()





profiler.add_function(hc.hierarchical_clustering)
profiler.add_function(hc._calculate_distance_clusters_min_distance)
profiler.add_function(hc.Cluster.calculate_distance)
#profiler.add_function(ejercicio1.resolve_test)
wrapper = profiler(preprocess.main)
wrapper()


profiler.print_stats(open("line_profiler","w"), output_unit=1)