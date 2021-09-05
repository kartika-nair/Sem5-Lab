"""
You can create any other helper funtions.
Do not modify the given functions
"""

import sys
import queue

def A_star_helper(priority, visited, goals, path, start_point, parents, cost, heuristic, minimum_cost):
    '''
    HELPER FUNCTION FOR A STAR
    '''

    while(priority.qsize()):
        t = priority.get()
        node = t[1]
        if node in visited:
            continue

        if node in goals:
            path = [start_point]
            if start_point == node:
                return path
            path = [node]
            node1 = node
            while not node1 == start_point:
                node1 = parents[node1]
                path.insert(0, node1)
            return path

        visited.add(node)

        for child in range(1, len(cost[node])):
            if cost[node][child] in [0, -1] or child in visited:
                continue

            temp_total = cost[node][child] + minimum_cost[node]
            if minimum_cost[child] == infinity or temp_total <= minimum_cost[child]:
                minimum_cost[child] = temp_total
                parents[child] = node

            priority.put((minimum_cost[child] + (0 if child >= len(heuristic) else heuristic[child]), child))


def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """

    infinity = sys.maxsize
    # print(infinity)

    path = []

    # TODO

    priority = queue.PriorityQueue()
    priority.put((0, start_point))
    minimum_cost = [infinity for i in range(len(cost))]

    parents = {start_point: start_point}
    visited = set()

    # while(priority.qsize()):
    #     t = priority.get()
    #     node = t[1]
    #     if node in visited:
    #         continue

    #     if node in goals:
    #         path = [start_point]
    #         if start_point == node:
    #             return path
    #         path = [node]
    #         node1 = node
    #         while not node1 == start_point:
    #             node1 = parents[node1]
    #             path.insert(0, node1)
    #         return path

    #     visited.add(node)

    #     for child in range(1, len(cost[node])):
    #         if cost[node][child] in [0, -1] or child in visited:
    #             continue

    #         temp_total = cost[node][child] + minimum_cost[node]
    #         if minimum_cost[child] == infinity or temp_total <= minimum_cost[child]:
    #             minimum_cost[child] = temp_total
    #             parents[child] = node

    #         priority.put((minimum_cost[child] + (0 if child >= len(heuristic) else heuristic[child]), child))

    path = A_star_helper(priority, visited, goals, path, start_point, parents, cost, heuristic, minimum_cost)

    # DONE

    return path


def DFS_helper(stack, path, visited, goals, start_point, parents, cost):
    '''
    HELPER FUNCTION FOR DFS
    '''

    while(stack.qsize()):
        node, path = stack.get()

        if visited[node] == 0:
            visited[node] = 1
        else:
            continue
        if node in goals:
            path = [node]
            if start_point == node:
                return path
            cur = node
            while not cur == start_point:
                cur = parents[cur]
                path.insert(0, cur)
            return path

        else:
            for i in range(len(cost[node]) - 1, 0, -1):
                if visited[i] or cost[node][i] in [0, -1]:
                    continue
                parents[i] = node
                stack.put((i, path))


def DFS_Traversal(cost, start_point, goals):
  
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """

    path = []

    # TODO

    stack = queue.LifoQueue(maxsize = 0)

    n = len(cost)
    visited = [0 for i in range(n)]
    parents = {}
    stack.put((start_point, [start_point]))

    # while(stack.qsize()):
    #     node, path = stack.get()

    #     if visited[node] == 0:
    #         visited[node] = 1
    #     else:
    #         continue
    #     if node in goals:
    #         path = [node]
    #         if start_point == node:
    #             return path
    #         cur = node
    #         while not cur == start_point:
    #             cur = parents[cur]
    #             path.insert(0, cur)
    #         return path

    #     else:
    #         for i in range(len(cost[node]) - 1, 0, -1):
    #             if visited[i] or cost[node][i] in [0, -1]:
    #                 continue
    #             parents[i] = node
    #             stack.put((i, path))

    path = DFS_helper(stack, path, visited, goals, start_point, parents, cost)

    # DONE

    return path
    

'''
cost = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1],
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
heuristic = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
start = 1
goals = [6, 7, 10]
print(DFS_Traversal(cost,start, goals))
'''
