def print_tree(mcts_obj, root_node):
    fifo = []
    for child_node in root_node.children:
        q = 0
        if mcts_obj.visits[child_node] > 0:
            q = mcts_obj.Q[child_node] / mcts_obj.visits[child_node]
        print(
            child_node,
            mcts_obj.Q[child_node],
            mcts_obj.visits[child_node],
            q
        )
        fifo.append(child_node)
    print("-")
    while fifo:
        root_node = fifo.pop()
        print("(", root_node, ")")
        for child_node in root_node.children:
            q = 0
            if mcts_obj.visits[child_node] > 0:
                q = mcts_obj.Q[child_node] / mcts_obj.visits[child_node]
            print(
                child_node,
                mcts_obj.Q[child_node],
                mcts_obj.visits[child_node],
                q
            )
            fifo.append(child_node)
        if root_node.children:
            print(" ")


def gen_tree_graph(root_node, G):
    fifo = []
    root_id = str(root_node)
    G.add_node(root_id)
    for child_node in root_node.children:
        child_id = str(child_node)
        G.add_node(child_id)
        G.add_edge(root_id, child_id)
        fifo.append(child_node)
    while fifo:
        root_node = fifo.pop()
        root_id = str(root_node)
        for child_node in root_node.children:
            child_id = str(child_node)
            G.add_node(child_id)
            G.add_edge(root_id, child_id)
            fifo.append(child_node)
