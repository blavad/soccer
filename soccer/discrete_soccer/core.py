
class Team(object):
    
    def __init__(self, nb_players=1):
        self.player = [Player(self) for i in range(nb_players)]
        
    def __len__(self):
        return len(self.player)
    
    @property
    def has_ball(self):
        for pl in self.player:
            if pl.has_ball:
                return True
        return False

class Team1(Team):
    
    def __init__(self, nb_players=1):
        super(Team1, self).__init__(nb_players)
        
    def init_config(self, w, h):
        for i, pl in enumerate(self.player):
            pl.has_ball = False
            if i==0:
                pl.pos = (h//2,0)
            if i==1:
                pl.pos = (0,0)
            if i==2:
                pl.pos = (h-1,0)
        return self

class Team2(Team):
    
    def __init__(self, nb_players=1):
        super(Team2, self).__init__(nb_players)
        
    def init_config(self, w, h):
        for i, pl in enumerate(self.player):
            pl.has_ball = False
            if i==0:
                pl.pos = (h//2,w-1)
            if i==1:
                pl.pos = (0,w-1)
            if i==2:
                pl.pos = (h-1,w-1)
        return self
        
        
class Player(object):
    def __init__(self, team, x=0, y=0):
        self.has_ball = False
        self.pos = (x,y)
        self.old_pos = self.pos
        self.team = team