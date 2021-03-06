"""This code generates interactive HTML file with MCTS Tree Visualized"""
import os

from monte_carlo_tree_search.trees.abstract_tree import TreeNode

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PREAMBLE_FILE = os.path.join(DATA_DIR, 'preamble')
POSTAMBLE_FILE = os.path.join(DATA_DIR, 'postamble')

class TreeVisualizer:

    def __init__(self, show_unvisited_nodes = False, level_interval = 3):
        self.show_unvisited_nodes = show_unvisited_nodes
        self.root_color = ["cyan"]
        self.list_of_colors = ["lime", "red"]
        self.nodes_str = ''
        self.edges_str = ''
        self.level_str = ''
        self.nodes_to_id = {}
        self.last_nod_id = 0
        self.states_as_dicts = ''
        self.level_interval = level_interval


    def add_node_to_dict(self, node):
        self.nodes_to_id[node] = self.last_nod_id
        self.last_nod_id += 1

    def generate_html(self, root, file_name):
        code = self.generate_tree_data(root)
        self.combine_html(code, file_name)
        self.__init__(show_unvisited_nodes=self.show_unvisited_nodes)

    def node_to_string(self, node: TreeNode):
        value_to_show_raw = node.value_acc.get()
        value_to_show = round(value_to_show_raw, 2) if value_to_show_raw is not None else value_to_show_raw
        conf_to_show = round(node.value_acc.get_confidence(), 2)
        return 'nodes.push(' + '{' + ' id: {}, label: \"Id: {} \\n V: {} \\n C: {} \\n CF: {} \"'.format(self.nodes_to_id[node], self.nodes_to_id[node],
                                                                                            value_to_show,
                                                                                            node.value_acc.count(),
                                                                                                         conf_to_show) + '}); \n'

    def node_to_level(self, node: TreeNode):
        return 'nodes[{}]["level"] = {}; \n'.format(self.nodes_to_id[node], self.level_interval*node.generation)

    def generate_tree_data(self, root: TreeNode):
        all_code = ''
        # BFS
        kiu = [root]
        self.add_node_to_dict(root)
        self.states_as_dicts += '<br>' + ' --- P = [{}, {}] ---'.format(root.state.active_players_hand().number_of_my_points(), root.state.previous_players_hand().number_of_my_points()) +\
                                str(self.nodes_to_id[root]) + str(root.observation.observation_dict) + '<br><br><br>'


        while len(kiu) > 0:
            # take first:
            #print(kiu)
            node_to_eval = kiu.pop(0)
            if node_to_eval.value_acc.count() > 0 or self.show_unvisited_nodes:
                self.nodes_str += self.node_to_string(node_to_eval)
                self.level_str += self.node_to_level(node_to_eval)
                for child in node_to_eval.children:
                    if child.value_acc.count() > 0 or self.show_unvisited_nodes:
                        kiu.append(child)
                        self.add_node_to_dict(child)
                        self.states_as_dicts += str(self.nodes_to_id[child]) + ' --- P = [{}, {}]'.format(child.recreate_state().active_players_hand().number_of_my_points(), child.recreate_state().previous_players_hand().number_of_my_points()) + str(child.observation.observation_dict) + '<br><br><br>'
                        self.edges_str += self.edge_to_string(node_to_eval, child)


    def edge_to_string(self, node1, node2):
        edge_caption = str(node2.parent_action.short_description())

        return 'edges.push({' +'from: {}, to: {}'.format(self.nodes_to_id[node1], self.nodes_to_id[node2])+ ',label: \"' + edge_caption + '\", ' +  'font: { align: "middle" } }); \n'

    def combine_html(self, tree_code, file_name):
        with open(PREAMBLE_FILE, 'r') as file:
            preamble = file.read()
        with open(POSTAMBLE_FILE, 'r') as file:
            postamble = file.read()

        combined = preamble + self.nodes_str + self.edges_str + self.level_str + postamble + self.states_as_dicts + '</body></html>'

        text_file = open(file_name, "w")
        text_file.write(combined)
        text_file.close()
