"""
Discret soccer game.
"""
import os
import cv2
import math
import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

import soccer
from soccer import BaseSoccerEnv
from soccer.core import Team1, Team2


def are_valide_coord(arr, x, y):
    return (x > 0 and x < arr.shape[1] and y > 0 and y < arr.shape[0])


def putImg(back, img, x, y, w, h=None, rate = 1.0):
    if h is None :
        h = w
    img2 = cv2.resize(img, (w,h))
    for idx, nx in enumerate(range(x, x+w)):
        for idy, ny in enumerate(range(y, y + h)):
            if img2[idy, idx, 3] > 250 and are_valide_coord(back, nx, ny):
                back[ny, nx, :] = rate*img2[idy, idx, :] + (1-rate)*back[ny, nx, :]
    return back

class ContinuousSoccerEnv(BaseSoccerEnv):
    """
    Description:
        Soccer game.
    Observation:
        Type: Discrete(Width x Height x NbAgent)
        Num	Observation                                 
        
    Actions:
        Type: Discrete(5)
        Num	Action
        0	Do nothing
        1	Front
        2	Back
        3	Left
        4	Right

    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    actions = [
        'none',
        'front',
        'back',
        'left',
        'right'
    ]
    
    act_types = ['discrete',
                 'continuous']
    
    obs_types = ['positions',
                 'image']


    def __init__(self, width_field=300, height_field=200, height_goal=None, nb_pl_team1=1, nb_pl_team2=1, act_type='discrete', obs_type='positions'):
        BaseSoccerEnv.__init__(self, width=width_field, height=height_field,height_goal=height_goal,nb_pl_team1=nb_pl_team1,nb_pl_team2=nb_pl_team2, type_config="continuous")
        
        
        # Ball 
        self.ball_pos = [np.random.randint(self.size_ball, self.height-self.size_ball), self.width//2-self.size_ball//2]
        
        # Autres parametres d etats
        assert act_type in ContinuousSoccerEnv.act_types
        self.act_type = act_type

        assert obs_type in ContinuousSoccerEnv.obs_types
        self.obs_type = obs_type
        
        self.speed_pl = 8
        self.ep_goal = max(4, self.width//50)

        self.velocity_ball = [0,0]

        self.action_space = spaces.Discrete(len(ContinuousSoccerEnv.actions))
        if obs_type is 'positions':
            self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 5+2*(self.n_players-1)))
        else :
            self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 64, 64))
             
        if self.act_type is 'discrete':
            self.action_space = spaces.Discrete(len(ContinuousSoccerEnv.actions))
        else :
            self.observation_space = spaces.Box(low=0, high=1, shape=(len(ContinuousSoccerEnv.actions)))
    
    
    def init_assets(self):

        u_j1 = os.path.join(os.path.dirname(soccer.__file__),'assets/j1_t.png')
        u_j2 = os.path.join(os.path.dirname(soccer.__file__),'assets/j2_t.png')
        u_ball = os.path.join(os.path.dirname(soccer.__file__),'assets/ball.png')

        self.j1 = cv2.cvtColor(cv2.resize(cv2.imread(u_j1, cv2.IMREAD_UNCHANGED), (self.size_player_w,self.size_player)), cv2.COLOR_BGRA2RGBA)
        self.j2 = cv2.cvtColor(cv2.resize(cv2.imread(u_j2, cv2.IMREAD_UNCHANGED), (self.size_player_w,self.size_player)), cv2.COLOR_BGRA2RGBA)
        
        self.ball = cv2.cvtColor(cv2.resize(cv2.imread(u_ball, cv2.IMREAD_UNCHANGED), (self.size_ball,self.size_ball)), cv2.COLOR_BGRA2RGBA)
        
    def diff_pos(self, pos_ref, pos_comp):
        diff = np.array(list(pos_comp)) - np.array(list(pos_ref))
        return tuple(diff)
        
        
    @property
    def state(self):
        if self.obs_type is "positions":
            states = []
            for me in self.all_players:
                obs = []#[me.pos[0]/self.height, me.pos[1]/self.width]
                b0 = self.diff_pos(me.pos, self.ball_pos)[0]/self.height if me.team is self.team1 else self.diff_pos(self.ball_pos, me.pos)[0]/self.height
                b1 = self.diff_pos(me.pos, self.ball_pos)[1]/self.width if me.team is self.team1 else self.diff_pos(self.ball_pos, me.pos)[1]/self.width
                g0 = self.diff_pos(me.pos, (self.goal_pos[0]+self.h_goal//2, self.width))[0]/self.height if me.team is self.team1 else self.diff_pos((self.goal_pos[0]+self.h_goal//2, 0), me.pos)[0]/self.height
                g1 = self.diff_pos(me.pos, (self.goal_pos[0]+self.h_goal//2, self.width))[1]/self.width if me.team is self.team1 else self.diff_pos((self.goal_pos[0]+self.h_goal//2, 0), me.pos)[1]/self.width
                # my_g0 = self.diff_pos(me.pos, (self.goal_pos[0]+self.h_goal//2, self.width))[0]/self.height if me.team is self.team1 else self.diff_pos((self.goal_pos[0]+self.h_goal//2, 0), me.pos)[0]/self.height
                my_g1 = self.diff_pos(me.pos, (self.goal_pos[0]+self.h_goal//2, 0))[1]/self.width if me.team is self.team1 else self.diff_pos((self.goal_pos[0]+self.h_goal//2, self.width), me.pos)[1]/self.width
                obs += [b0,b1,g0,g1, my_g1]
                for pl_w_me in me.team.player:
                    if pl_w_me is not me:
                        o0 = self.diff_pos(me.pos, pl_w_me.pos)[0]/self.height if me.team is self.team1 else self.diff_pos(pl_w_me.pos, me.pos)[0]/self.height
                        o1 = self.diff_pos(me.pos, pl_w_me.pos)[1]/self.width if me.team is self.team1 else self.diff_pos(pl_w_me.pos, me.pos)[1]/self.width
                        obs += [o0,o1]
                for _, pl in enumerate(self.all_players):
                    if pl.team is not me.team:
                        o0 = self.diff_pos(me.pos, pl.pos)[0]/self.height if me.team is self.team1 else self.diff_pos(pl.pos, me.pos)[0]/self.height
                        o1 = self.diff_pos(me.pos, pl.pos)[1]/self.width if me.team is self.team1 else self.diff_pos(pl.pos, me.pos)[1]/self.width
                        obs += [o0,o1]
                states += [obs]
            return states
        else:
            return self.renderGame()

    def new_pos(self, player, action):
        l_pos = list(player.pos)
        if isinstance(player.team, Team1):
            l_pos[1] += self.speed_pl if action=='front' and l_pos[1] + self.size_player_w + self.speed_pl < self.width else 0
            l_pos[1] -= self.speed_pl if action=='back' and l_pos[1] - self.speed_pl > 0 else 0
            l_pos[0] += self.speed_pl if action=='right' and l_pos[0] + self.speed_pl + self.size_player < self.height else 0
            l_pos[0] -= self.speed_pl if action=='left' and l_pos[0] - self.speed_pl > 0 else 0
        if isinstance(player.team, Team2):
            l_pos[1] += self.speed_pl if action=='back' and l_pos[1] + self.size_player_w + self.speed_pl < self.width else 0
            l_pos[1] -= self.speed_pl if action=='front' and l_pos[1] - self.speed_pl > 0 else 0
            l_pos[0] += self.speed_pl if action=='left' and l_pos[0] + self.speed_pl + self.size_player < self.height else 0
            l_pos[0] -= self.speed_pl if action=='right' and l_pos[0] - self.speed_pl > 0 else 0
        return tuple(l_pos)

    def reward(self, action=None):
        rew_team1 = 0
        rew_team2 = 0
        but = self.buuut()
        if but != [0,0]:
            self.done_flag = True
            self.score += but
        rew_team1 = rew_team1 + (but[0] - but[1]) 
        rew_team2 = rew_team2 + (but[1] - but[0])
        rew = [rew_team1]*len(self.team1) + [rew_team2]*len(self.team2)
        done = [self.done_flag]*self.n_players
        return rew, done
            
    def buuut(self):
        if self.ball_pos[1]+self.size_ball >= self.width-self.ep_goal and self.ball_pos[0] >= self.goal_pos[0] and self.ball_pos[0] < self.goal_pos[1]: 
            return [1,0]
        
        if self.ball_pos[1] <= self.ep_goal and self.ball_pos[0] >= self.goal_pos[0] and self.ball_pos[0] < self.goal_pos[1]: 
            return [0,1]
        return [0,0]

    def update_field(self):
        pass

    def collision_pl(self, p1,p2):
        return (p1.pos[1] < p2.pos[1] + self.size_player_w and
            p1.pos[1] + self.size_player_w > p2.pos[1] and
            p1.pos[0] < p2.pos[0] + self.size_player and
            p1.pos[0] + self.size_player > p2.pos[0])

    def collision_ball(self, pl):
        return (self.ball_pos[1] < pl.pos[1] + self.size_player_w and
                self.ball_pos[1] + self.size_ball > pl.pos[1] and
                self.ball_pos[0] < pl.pos[0] + self.size_player and
                self.ball_pos[0] + self.size_ball > pl.pos[0])

    def gere_conflits(self, p1, p2):
        # p1 vers la droite
        if p1.pos[1] - p1.old_pos[1] > 0:
            # p2 vers la gauche
            if p2.pos[1] - p2.old_pos[1] < 0:
                p1.pos = p1.old_pos
                p2.pos = p2.old_pos
            # p2 vers le haut
            elif p2.pos[0] - p2.old_pos[0] < 0:
                if p2.old_pos[0] < p1.pos[0]+self.size_player:
                    p2.pos = (p2.pos[0], p1.pos[1]+self.size_player_w)
                else:
                    p1.pos = (p2.pos[0]-self.size_player, p1.pos[1])
            # p2 vers le bas
            elif p2.pos[0] - p2.old_pos[0] > 0 :
                if p2.old_pos[0]+self.size_player > p1.pos[0]:
                    p2.pos = (p2.pos[0], p1.pos[1]+self.size_player_w)
                else:
                    p1.pos = (p2.pos[0]+self.size_player, p1.pos[1])
            # p2 ne bouge pas
            elif p2.pos[0] - p2.old_pos[0] == 0 and p2.pos[1] - p2.old_pos[1] == 0:
                p2.pos = (p2.pos[0], p1.pos[1]+self.size_player_w)
                
        # p1 vers le haut        
        elif p1.pos[0] - p1.old_pos[0] < 0:
            # p2 vers le bas
            if p2.pos[0] - p2.old_pos[0] > 0:
                p1.pos = p1.old_pos
                p2.pos = p2.old_pos   
            # p2 vers droite, gauche ou rien
            if p1.old_pos[0] > p2.pos[0]+self.size_player:
                p2.pos = (p1.pos[0]-self.size_player, p2.pos[1])
        
        # p1 vers le bas        
        elif p1.pos[0] - p1.old_pos[0] < 0:
            # p2 vers le haut
            if p2.pos[0] - p2.old_pos[0] < 0:
                p1.pos = p1.old_pos
                p2.pos = p2.old_pos   
            # p2 vers droite, gauche ou rien
            if p1.old_pos[0] < p2.pos[0]-self.size_player:
                p2.pos = (p1.pos[0]+self.size_player, p2.pos[1])
              
        # p1 vers la gauche     
        elif p1.pos[1] - p1.old_pos[1] < 0:
            # p2 vers le haut
            if p2.pos[0] - p2.old_pos[0] < 0:
                p2.pos = (p2.pos[0], p1.pos[1]-self.size_player_w)
            # p2 vers le bas
            if p2.pos[0] - p2.old_pos[0] > 0:
                p2.pos = (p2.pos[0], p1.pos[1]-self.size_player_w)
            
            # p2 ne bouge pas
            elif p2.pos[0] - p2.old_pos[0] == 0 and p2.pos[1] - p2.old_pos[1] == 0:
                p2.pos = (p2.pos[0], p1.pos[1] - self.size_player_w)
    
    def gere_conflits_ball(self, pl):
        vel = [0,0]
        # p1 vers la droite
        if pl.pos[1] - pl.old_pos[1] > 0:
            vel = [0, self.speed_pl]    
            
        # p1 vers la gauche
        if pl.pos[1] - pl.old_pos[1] < 0:
            vel = [0,-self.speed_pl]    
            
        # p1 vers la bas
        if pl.pos[0] - pl.old_pos[0] > 0:
            vel = [self.speed_pl,0]    
            
        # p1 vers la droite
        if pl.pos[0] - pl.old_pos[0] < 0:
            vel = [ -self.speed_pl,0]    
        self.velocity_ball[0] += vel[0]
        self.velocity_ball[1] += vel[1]
    
    def is_valide_pos(self, pos0, pos1, w, h):
        return pos0>0 and pos0+h<self.height and pos1>0 and pos1+w<self.width
    
    def update_state(self, actions):
        for i, (pl, act) in enumerate(list(zip(self.all_players, actions))):
            pl.pos = self.new_pos(pl, act)
            if pl.pos == pl.old_pos:
                actions[i] = 'none'

        for pl1, act1 in list(zip(self.all_players, actions)):
            for pl2, act2 in list(zip(self.all_players, actions)):
                if pl1 is not pl2:
                    if self.collision_pl(pl1,pl2):
                        self.gere_conflits(pl1,pl2)

        self.velocity_ball[0] //= 2
        self.velocity_ball[1] //= 2

        for pl in self.all_players:
            if self.collision_ball(pl):
                self.velocity_ball = [0,0]
                self.gere_conflits_ball(pl)
                
        
                    
        self.ball_pos[0] += self.velocity_ball[0] if self.is_valide_pos(self.ball_pos[0]+self.velocity_ball[0], self.ball_pos[1],self.size_ball, self.size_ball) else 0
        self.ball_pos[1] += self.velocity_ball[1] if self.is_valide_pos(self.ball_pos[0], self.ball_pos[1]+self.velocity_ball[1],self.size_ball, self.size_ball) else 0
        
        for p in self.all_players:
            if self.is_valide_pos(p.pos[0], p.pos[1], self.size_player_w, self.size_player):
                p.old_pos = p.pos
            else:
                p.pos = p.old_pos
                                
    def draw_goal(self, img):
        ep = self.ep_goal
        y_deb = self.goal_pos[0]
        y_fin = self.goal_pos[1] 
        but1_img = np.zeros((y_fin-y_deb, ep, 4))[:,:,:4] = [50,50,150,255]
        but2_img = np.zeros((y_fin-y_deb, ep, 4))[:,:,:4] = [150,50,50,255]
        img[y_deb:y_fin, 0:ep] = but1_img
        img[y_deb:y_fin, self.width-ep:] = but2_img
        return img
        
    def draw_ball(self, img):
        x_offset = self.ball_pos[1]
        y_offset = self.ball_pos[0] 
        ball_img = self.ball
        ind = np.where(ball_img[:,:,3]>250)
        img[ind[0]+y_offset, ind[1]+x_offset] = ball_img[ind]
        return img        

    def draw_player(self, img, p):
        
        x_offset = p.pos[1]
        y_offset = p.pos[0] 
        player_img = self.j1 if isinstance(p.team, Team1) else self.j2
        ind = np.where(player_img[:,:,3]>250)
        img[ind[0]+y_offset, ind[1]+x_offset] = player_img[ind]
        return img