from rdflib.graph import Graph
from lib.converters import n3_to_nx, digraph_to_graph, nx_to_n3
from lib.helpers import prepare, add_generalization_predicates
from lib.core import nx_pagerank, shrink_py_pr


def netsdm_reduce(input_dict):
    data = Graph()
    prepare(data)
    data.parse(data=input_dict['examples'], format='n3')
    for ontology in input_dict['bk_file']:
        data.parse(data=ontology, format='n3')
    full_network, positive_nodes, generalization_predicates = n3_to_nx(data, input_dict['target'])
    if not input_dict['directed']:
        full_network = digraph_to_graph(full_network)
    node_list = full_network.nodes()
    scores, scores_dict = nx_pagerank(full_network, node_list, positive_nodes)
    shrink_py_pr(full_network, node_list, scores, float(input_dict['minimum_ranking']), positive_nodes)

    rdf_network = nx_to_n3(full_network)
    add_generalization_predicates(rdf_network, generalization_predicates)

    return { 'bk_file': rdf_network.serialize(format='n3') }
