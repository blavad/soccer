"""
Discret soccer game.
"""
import soccer
import math
import os
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import cv2
from .core import Team1, Team2

class DiscreteSoccerEnv(gym.Env):
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
    
    obs_types = ['integer',
                 'matrix']

    score = np.array([0,0])

    l_bound = 100

    def __init__(self, width_field=5, height_field=4, height_goal=None, nb_pl_team1=1, nb_pl_team2=1, obs_type='integer'):

        DiscreteSoccerEnv.score = np.array([0,0])
        
        # Field parameters
        self.w_field = width_field
        self.h_field = height_field
        self.h_goal = self.h_field//2 if height_goal is None else height_goal
        self.goal_pos = (self.h_field//2 - self.h_goal//2, self.h_field//2 + (self.h_goal-self.h_goal//2))
        self.field = np.zeros((self.h_field, self.w_field))
        
        # Dimensions
        self.width = width_field*DiscreteSoccerEnv.l_bound
        self.height = height_field*DiscreteSoccerEnv.l_bound
        
        # Players parameters
        self.team = [Team1(nb_pl_team1).init_config(self.w_field, self.h_field), Team2(nb_pl_team2).init_config(self.w_field, self.h_field)]
        self.update_field()

        # Autres parametres d etats
        assert obs_type in DiscreteSoccerEnv.obs_types
        self.obs_type = obs_type
        self.done_flag = False

        self.action_space = spaces.Discrete(len(DiscreteSoccerEnv.actions))
        if obs_type is 'integer':
            self.observation_space = spaces.Discrete(self.state_space)
        else :
            self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.h_field, self.w_field),
            dtype=np.uint8)
            
        self.init_assets()
            
        self.viewer = None

    def init_assets(self):
        c = DiscreteSoccerEnv.l_bound

        u_j1 = os.path.join(os.path.dirname(soccer.__file__),'discrete_soccer/assets/j1.png')
        u_j1b = os.path.join(os.path.dirname(soccer.__file__),'discrete_soccer/assets/j1_ball.png')
        u_j2 = os.path.join(os.path.dirname(soccer.__file__),'discrete_soccer/assets/j2.png')
        u_j2b = os.path.join(os.path.dirname(soccer.__file__),'discrete_soccer/assets/j2_ball.png')

        self.j1 = cv2.cvtColor(cv2.resize(cv2.imread(u_j1), (c,c)), cv2.COLOR_BGR2RGB)
        self.j1_ball = cv2.cvtColor(cv2.resize(cv2.imread(u_j1b), (c,c)), cv2.COLOR_BGR2RGB)
        self.j2 = cv2.cvtColor(cv2.resize(cv2.imread(u_j2), (c,c)), cv2.COLOR_BGR2RGB)
        self.j2_ball = cv2.cvtColor(cv2.resize(cv2.imread(u_j2b), (c,c)), cv2.COLOR_BGR2RGB)

    @property
    def state(self):
        if self.obs_type is 'integer':
            return self.calculate_int_state()
        else:
            return self.map_state()
        
    @property
    def team1(self):
        return self.team[0]
    
    @property
    def team2(self):
        return self.team[1]
    
    @property
    def n_players(self):
        return len(self.team1) + len(self.team2)
    
    @property
    def state_space(self):
        return (self.n_players)*(self.w_field*self.h_field)**(self.n_players)
    
    @property
    def all_players(self):
        return self.team1.player + self.team2.player
        
    def pl_state(self, i):
        pl_pos = self.all_players[i].pos
        return pl_pos[0] + self.h_field * pl_pos[1]
    
    def calculate_int_state(self):
        coef = (self.w_field*self.h_field)**np.arange(self.n_players)
        pos_pl = np.array([self.pl_state(i) for i in range(self.n_players)])
        tmp_state = sum(coef*pos_pl)
        return tmp_state
        
    def reset(self):
        self.team[0] = self.team[0].init_config(self.w_field, self.h_field)
        self.team[1] = self.team[1].init_config(self.w_field, self.h_field)
        self.done_flag = False
        self.update_field()
        return [self.state]*self.n_players

    def step(self, actions):
        action = []
        try :
            actions = list(actions)
        except TypeError :
            actions = [actions]          
        for act in actions:
            assert self.action_space.contains(act), "%r (%s) invalid" % (act, type(act))
            action += [DiscreteSoccerEnv.actions[act]]
        if len(action) < self.n_players:
            action += 'none'*(self.n_players- len(action))
        rew, done = self.reward(action)
        
        self.update_state(action)
        
        self.update_field()
        
        return [self.state]*self.n_players, rew, done, {}

    def new_pos(self, player, action):
        l_pos = list(player.pos)
        if isinstance(player.team, Team1):
            l_pos[1] += 1 if action=='front' and l_pos[1]+1 < self.w_field else 0
            l_pos[1] -= 1 if action=='back' and l_pos[1] > 0 else 0
            l_pos[0] += 1 if action=='right' and l_pos[0]+1 < self.h_field else 0
            l_pos[0] -= 1 if action=='left' and l_pos[0] > 0 else 0
        if isinstance(player.team, Team2):
            l_pos[1] += 1 if action=='back' and l_pos[1]+1 < self.w_field else 0
            l_pos[1] -= 1 if action=='front' and l_pos[1] > 0 else 0
            l_pos[0] += 1 if action=='left' and l_pos[0]+1 < self.h_field else 0
            l_pos[0] -= 1 if action=='right' and l_pos[0] > 0 else 0
        return tuple(l_pos)

    def reward(self, action):
        rew_team1 = 0
        rew_team2 = 0
        for pl, act in list(zip(self.all_players, action)):
            but = self.buuut(pl, act)
            if but != [0,0]:
                self.done_flag = True
                DiscreteSoccerEnv.score += but
            rew_team1 = rew_team1 + (but[0] - but[1])*1
            rew_team2 = rew_team2 + (but[1] - but[0])*1
        # rew_team1 += int(self.team1.has_ball) - int(self.team2.has_ball)
        # rew_team2 += int(self.team2.has_ball) - int(self.team1.has_ball)
        rew = [rew_team1]*len(self.team1) + [rew_team2]*len(self.team2)
        done = [self.done_flag]*self.n_players
        return rew, done
            
    def buuut(self, pl, action):
        if action=='front':
            if isinstance(pl.team, Team1) and pl.has_ball and pl.pos[1]+1 >= self.w_field and pl.pos[0] >= self.goal_pos[0] and pl.pos[0] < self.goal_pos[1]: 
                pl.has_ball = False
                return [1,0]
            
            if isinstance(pl.team, Team2) and pl.has_ball and pl.pos[1] < 1 and pl.pos[0] >= self.goal_pos[0] and pl.pos[0] < self.goal_pos[1]:
                pl.has_ball = False
                return [0,1]
        return [0,0]

    def update_field(self):
        self.field = np.zeros((self.h_field, self.w_field))
        for i, pl in enumerate(self.all_players):
            self.field[pl.pos] = 10*(i+1) if pl.has_ball else i+1

    def update_state(self, actions):
        for i, (pl, act) in enumerate(list(zip(self.all_players, actions))):
            # print(pl.pos, ' - ', act)
            pl.pos = self.new_pos(pl, act)
            if pl.pos == pl.old_pos:
                actions[i] = 'none'
        
        conflit = {}
        for pl, act in list(zip(self.all_players, actions)):
            if pl.pos in conflit.keys():
                conflit[pl.pos] += [[pl,act]]  
            else:
                conflit[pl.pos] = [[pl,act]] 
                
        
        self.gere_conflits(conflit)
        
        # print('Conflits avant step ',conflit)
        
        for p in self.all_players:
            p.old_pos = p.pos
        
    
                
    def gere_conflits(self, conflit):
        # Update major conflicts
        for conf_pos, conf_pl in conflit.items():
            if len(conf_pl) >1:
                if not 'none' in list(zip(*conf_pl))[1]:
                    num_pl = int(len(conf_pl)*np.random.random())
                    for i, p in enumerate(conf_pl):
                        if i != num_pl:
                            p[0].pos = p[0].old_pos
                            if p[0].has_ball:
                                p[0].has_ball = False
                                conf_pl[num_pl][0].has_ball = True
                                
                                
                            # if p[0].pos in conflit.keys():
                            #     conflit[pl.pos] += [[pl,'none']]  
                            # else:
                            #     conflit[pl.pos] = [[pl,'none']] 
                            # conf_pl.remove(p)
        
        for conf_pos, conf_pl in conflit.items():
            if len(conf_pl) >1:
                if 'none' in list(zip(*conf_pl))[1]:
                    pl_stay = list(zip(*conf_pl))[1].index('none')
                    for i, p in enumerate(conf_pl):
                        if i != pl_stay:
                            p[0].pos = p[0].old_pos
                            if p[0].has_ball:
                                keep_ball = np.random.random()<0.7
                                p[0].has_ball = keep_ball
                                conf_pl[pl_stay][0].has_ball = not keep_ball
                            elif conf_pl[pl_stay][0].has_ball:
                                keep_ball = np.random.random()<0.3
                                conf_pl[pl_stay][0].has_ball = keep_ball
                                p[0].has_ball = not keep_ball
                                
                                
                            # if p[0].pos in conflit.keys():
                            #     conflit[p[0].pos] += [[p,'none']]  
                            # else:
                            #     conflit[p[0].pos] = [[p,'none']] 
                            # conf_pl.remove(p)        
        
        
    ########## RENDER PART ##############
    
    def render(self, mode='human'):
        if mode == 'human':
            return self.render_human(mode)
        elif mode == 'rbg_array':
            return self.render_rgb_array()
        return self.render_array()

    def render_human(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer.imshow(self.render(mode='rbg_array'))

    def render_array(self):
        print(self.field)

    def render_rgb_array(self):    
        color = np.array([50,150,50])
        return np.concatenate((self.renderGame(color_background=color), self.renderInfos(score=DiscreteSoccerEnv.score,color_background=color-50)), axis=0)
        

    def renderGame(self, color_background=[50,200,50]):
        img = np.full(
            (self.height, self.width, 3),
            255,
            dtype=np.uint8,
        )
        img[:,:,:3] = color_background
        for p in self.all_players:
            img = self.draw_player(img, p)
        img = self.draw_goal(img)
        return img
        
    def renderInfos(self, score=None, color_background=[50,200,200]):
        height = self.height//6
        infosImg = np.full(
            (height, self.width, 3),
            255,
            dtype=np.uint8,
        )
        infosImg[:,:,:3] = color_background
        return self.displayInfos(infosImg, score)
         

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
            
    def displayInfos(self, img,  score):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0,0,0)
        cv2.putText(img, "Blue {} - {} Red".format(DiscreteSoccerEnv.score[0],DiscreteSoccerEnv.score[1]), (2*self.width//7, self.height//10), font, 1., color, 1, cv2.LINE_AA)
        # cv2.putText(img, "{}".format(DiscreteSoccerEnv.score[1]), (4*self.width//7, self.height//10), font, 1., color, 1, cv2.LINE_AA)
        return img

    def draw_goal(self, img):
        ep = max(4, DiscreteSoccerEnv.l_bound//10)
        y_deb = self.goal_pos[0]*DiscreteSoccerEnv.l_bound
        y_fin = self.goal_pos[1]*DiscreteSoccerEnv.l_bound 
        but1_img = np.zeros((y_fin-y_deb, ep, 3))[:,:,:3] = [50,50,150]
        but2_img = np.zeros((y_fin-y_deb, ep, 3))[:,:,:3] = [150,50,50]
        img[y_deb:y_fin, 0:ep] = but1_img
        img[y_deb:y_fin, self.width-ep:] = but2_img
        return img
        

    def draw_player(self, img, p):
        x_offset = p.pos[1]*DiscreteSoccerEnv.l_bound
        y_offset = p.pos[0]*DiscreteSoccerEnv.l_bound 
        if isinstance(p.team, Team1):
            player_img = self.j1 if not p.has_ball else self.j1_ball
        if isinstance(p.team, Team2):
            player_img = self.j2 if not p.has_ball else self.j2_ball
            
        img[y_offset:y_offset+player_img.shape[0], x_offset:x_offset+player_img.shape[1]] = player_img
        return img
    
    def map_state(self):
        tmp_state = np.zeros((3, self.h_field, self.w_field))
        for pl in self.team1.player:
            tmp_state[1, pl.pos[0],pl.pos[1]] = 1
            if pl.has_ball:
                tmp_state[0,pl.pos[0],pl.pos[1]] = 1
        for pl in self.team2.player:
            tmp_state[2, pl.pos[0],pl.pos[1]] = 1
            if pl.has_ball:
                tmp_state[0, pl.pos[0],pl.pos[1]] = 1
                
        return tmp_state
