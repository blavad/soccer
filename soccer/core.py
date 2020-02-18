
class Team(object):
    
    def __init__(self, nb_players=1):
        self.player = [Player(self) for i in range(nb_players)]
        
    def __len__(self):
        return len(self.player)
    
    def init_config(self, w, h, size_pl=20, type_config="discrete"):
        for i, pl in enumerate(self.player):
            pl.has_ball = False
            pl.pos = self._config(w,h,size_pl)[type_config][i]
        return self
    
    @property
    def has_ball(self):
        for pl in self.player:
            if pl.has_ball:
                return True
        return False

class Team1(Team):
    
    def __init__(self, nb_players=1):
        super(Team1, self).__init__(nb_players)
    
    def _config(self, w, h, size_pl):
        return {"discrete": [(h//2,0), (0,0), (h-1,0)],
                "continuous": [(h//2 - int(0.5*size_pl),int(2*size_pl)), (int(0.1*size_pl),int(0.5*size_pl)), (h-int(1.1*size_pl),int(0.5*size_pl))]}
        

class Team2(Team):
    
    def __init__(self, nb_players=1):
        super(Team2, self).__init__(nb_players)

    def _config(self, w, h, size_pl):
        return {"discrete": [(h//2,w-1), (0,w-1), (h-1,w-1)],
                "continuous": [(h//2 - int(0.5*size_pl), w-int(2.36*size_pl)), (int(0.1*size_pl), w-int(0.86*size_pl)), (h-int(1.1*size_pl), w-int(0.86*size_pl))]
                }
        
        
class Player(object):
    def __init__(self, team, x=0, y=0):
        self.has_ball = False
        self.pos = (x,y)
        self.old_pos = self.pos
        self.team = team
    
    @property
    def x(self):
        return self.pos[0]