from Representations.DagGP import DagLinear


class FlatTree(object):
    tree1 = {"root": ["up", "down", "left", "right", "pick", "drop"]}
    representation = None
    domain = None

    def __init__(self, domain, seed=1):
        self.domain = domain
        self.representation = DagLinear(domain, root="root", tree=self.tree1, terminals=self.terminals, seed=seed)

    def terminals(self, o, s):
        if o == "root":
            return self.domain.is_terminal(s)
        return False


class Tree1(object):
    tree1 = {"root": ["get", "put"],
             "get": ["pick", "navi_get"],
             "put": ["drop", "navi_put"],
             "navi_get": ["up", "down", "left", "right"],
             "navi_put": ["up", "down", "left", "right"]
            }
    representation = None
    domain = None

    def __init__(self, domain, seed=1):
        self.domain = domain
        self.representation = DagLinear(domain, root="root", tree=self.tree1, terminals=self.terminals, seed=seed)

    def terminals(self, o, s):
        s_dest = s[0]
        s_src = s[1]
        s_x = s[2]
        s_y = s[3]

        if o == "root":
            return self.domain.is_terminal(s)
        if o == "get":
            return s_src == self.domain.get_map_size()-1
        if o == "put":
            return s_dest == s_src
        if o == "navi_get":
            if s_src < self.domain.map_size-1:
                return self.domain.get_spot_locations()[s_src] == [s_x, s_y]
            else:
                return True
        if o == "navi_put":
            return self.domain.get_spot_locations()[s_dest] == [s_x, s_y]

        return False




