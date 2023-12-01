This repository contains all the codes used for doing the analysis and creating fogures for the following article.<br>
Kirtonia, S., Sun, Y., Chen, Z.-L. (in press), Selection of Auto-Carrier Loading Policy in Automobile Shipping. IISE Transactions.<br>
The following table shows how the results are produced for the benchmark problem and the full-scale analysis. The table also shows the code file used for generating each plot based on the data.

<b>Code and data used for producing the results and plots.</b>
<table width="828">
<tbody>
<tr>
<td width="72">
<p>Which results to reproduce</p>
</td>
<td width="282">
<p>Data File</p>
</td>
<td width="162">
<p>Code File</p>
</td>
<td width="192">
<p>Expected output</p>
</td>
<td width="120">
<p>Run time at the above-specified computer conditions</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 1</p>
</td>
<td colspan="4" width="756">
<p>Hand-drawn diagram</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 2</p>
</td>
<td colspan="4" width="756">
<p>Hand-drawn diagram</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 3</p>
</td>
<td colspan="4" width="756">
<p>Hand-drawn diagram</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 4</p>
</td>
<td colspan="4" width="756">
<p>Hand-drawn diagram</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 5</p>
</td>
<td colspan="4" width="756">
<p>Hand-drawn diagram</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 6</p>
</td>
<td colspan="4" width="756">
<p>Hand-drawn diagram</p>
</td>
</tr>
<tr>
<td width="72">
<p>Generate distance matrix</p>
</td>
<td width="282">
<p>files/given/locations_cleaned_fl.pkl</p>
<p>files/given/locations_cleaned.pkl</p>
</td>
<td width="162">
<p>0. generate_distance_matrix.py</p>
</td>
<td width="192">
<p>files/real_data/distance_matrix_fl_road.pkl</p>
<p>files/real_data/distance_matrix_road.pkl</p>
</td>
<td width="120">
<p>&nbsp;</p>
</td>
</tr>
<tr>
<td width="72">
<p>Benchmark Analysis</p>
</td>
<td width="282">
<p>files/given/locations_cleaned_fl.pkl</p>
</td>
<td width="162">
<p>Run sequentially</p>
<p>1.benchmark_initial_data_generation.py</p>
<p>2.benchmark_data_generation.py</p>
<p>3.benchmark_analysis.py</p>
</td>
<td width="192">
<p>files/solution/t_coef_100_exp_250_routes_6_slots_benchmark_same.pkl</p>
<p>files/solution/t_coef_100_exp_250_routes_6_slots_from_tampa_same. pkl</p>
<p>files/solution/t_coef_100_exp_250_routes_6_slots_to_tampa_same. pkl</p>
<p>&nbsp;</p>
</td>
<td width="120">
<p>Varies</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 7</p>
</td>
<td width="282">
<p>files/given/us states data shape/usa-states-census-2014.shp</p>
<p>files/given/cb_2018_us_cbsa_500k/cb_2018_us_cbsa_500k.shp</p>
<p>files/real_data/locations_cleaned_fl.pkl</p>
</td>
<td width="162">
<p>f7_metro_areas.py</p>
</td>
<td width="192">
<p>figs/figure_metro.svg</p>
</td>
<td width="120">
<p>1.24 s</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 8</p>
</td>
<td width="282">
<p>files/solution/t_coef_100_exp_250_routes_6_slots_benchmark_same. pkl</p>
<p>files/graph_info/p1a_graph_3_21_compare_p1a_p0.pickle</p>
<p>files/graph_info/p0_graph_new_3_21_compare_p1a_p0.pickle</p>
</td>
<td width="162">
<p>f8_compare_p0_1a.py</p>
</td>
<td width="192">
<p>figs/figure8a.svg</p>
<p>figs/figure8b.svg</p>
</td>
<td width="120">
<p>7.95 s</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 9</p>
</td>
<td width="282">
<p>files/solution/t_coef_100_exp_250_routes_6_slots_benchmark_same. pkl</p>
</td>
<td width="162">
<p>f9_compare_benchmark.py</p>
</td>
<td width="192">
<p>figs/figure9.svg</p>
</td>
<td width="120">
<p>6.9 s for fig 9,10,11</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 10</p>
</td>
<td width="282">
<p>files/solution/t_coef_100_exp_250_routes_6_slots_benchmark_same. pkl</p>
</td>
<td width="162">
<p>f9_compare_benchmark.py</p>
</td>
<td width="192">
<p>figs/figure10.svg</p>
</td>
<td width="120">
<p>&nbsp;</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 11</p>
</td>
<td width="282">
<p>files/solution/t_coef_100_exp_250_routes_6_slots_benchmark_same. pkl</p>
</td>
<td width="162">
<p>f9_compare_benchmark.py</p>
</td>
<td width="192">
<p>figs/figure11.svg</p>
</td>
<td width="120">
<p>&nbsp;</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 12</p>
</td>
<td width="282">
<p>files/graph_info/p1a_graph_3_25_slot4.pickle</p>
<p>files/graph_info/p1b_graph_3_28_slot4.pickle</p>
</td>
<td width="162">
<p>f12_suboptimality.py</p>
</td>
<td width="192">
<p>figs/figure12a.svg</p>
<p>figs/figure12b.svg</p>
</td>
<td width="120">
<p>1.46 s</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 13</p>
</td>
<td width="282">
<p>files/solution/t_coef_100_exp_250_routes_6_slots_benchmark_same. pkl</p>
</td>
<td width="162">
<p>f13_sens_reloading_cost.py</p>
</td>
<td width="192">
<p>figs/figure13.svg</p>
</td>
<td width="120">
<p>6.35 s</p>
</td>
</tr>
<tr>
<td width="72">
<p>&nbsp;</p>
<p>Figure 14</p>
</td>
<td width="282">
<p>files/organized/sensitivity n_route per experiment3.csv</p>
</td>
<td width="162">
<p>f14_sens_num_route_exp.py</p>
</td>
<td width="192">
<p>figs/figure14.svg</p>
</td>
<td width="120">
<p>0.37 s</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 15</p>
</td>
<td width="282">
<p>files/solution/t_coef_100_exp_250_routes_6_slots_benchmark_same.pkl</p>
<p>files/solution/t_coef_100_exp_250_routes_6_slots_to_tampa_same.pkl</p>
<p>files/solution/t_coef_100_exp_250_routes_6_slots_from_tampa_same.pkl</p>
</td>
<td width="162">
<p>f15_sesn_spatial_distribution.py</p>
</td>
<td width="192">
<p>figs/figure15.svg</p>
<p>&nbsp;</p>
</td>
<td width="120">
<p>0.42 s</p>
</td>
</tr>
<tr>
<td width="72">
<p>Table 1</p>
</td>
<td width="282">
<p>Data inside code file</p>
<p>files/real_data/distance_matrix_road.pkl</p>
<p>files/realistic/order_pick_drop_distance_real_45_orders.pkl</p>
</td>
<td width="162">
<p>Run sequentially.</p>
<p>realistic_0_initial_data.py</p>
<p>realistic_1_data_generation.py</p>
<p>realistic_2_computation.py</p>
<p>realistic_3_solve_optimization.py</p>
</td>
<td width="192">
<p>Table 1</p>
</td>
<td width="120">
<p>As reported in the tables</p>
</td>
</tr>
<tr>
<td width="72">
<p>Table 2</p>
</td>
<td width="282">
<p>Same as Table 1. Change the parameter to run</p>
</td>
<td width="162">
<p>&nbsp;</p>
</td>
<td width="192">
<p>&nbsp;</p>
</td>
<td width="120">
<p>&nbsp;</p>
</td>
</tr>
<tr>
<td width="72">
<p>Table 3</p>
</td>
<td width="282">
<p>Same as Table 1. Change the parameter to run</p>
</td>
<td width="162">
<p>&nbsp;</p>
</td>
<td width="192">
<p>&nbsp;</p>
</td>
<td width="120">
<p>&nbsp;</p>
</td>
</tr>
<tr>
<td width="72">
<p>Figure 16</p>
</td>
<td width="282">
<p>files/us states data shape/usa-states-census-2014.shp</p>
<p>files/realistic/order_pick_drop_distance_real_45_orders.pkl</p>
<p>files/real_data/locations_cleaned.pkl</p>
<p>files/realistic/v1/t_coef_summary_2_sets_800_r_25_a_45_1b_1c_0.pkl</p>
</td>
<td width="162">
<p>f16_order_od_pair.py</p>
</td>
<td width="192">
<p>figs/figure16.svg</p>
</td>
<td width="120">
<p>5.6 s</p>
</td>
</tr>
</tbody>
</table>

Note: Due to limitations on file size by GitHub, only the code files are uploaded along with the necessary data in the 'given' folder. All the files necessary to generate results and the plots can be generated by running the code files sequentially, as shown in the table. Some code files can generate multiple data files by themselves or by changing the parameters in the code file.
