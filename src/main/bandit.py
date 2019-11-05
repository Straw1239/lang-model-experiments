from normal_inverse_gamma import *
import torch

class ThompsonTuner():
    def __init__(self, stepper, num_options, device):
        self.stepper = stepper
        self.arms = num_options        
        self.belief = standardNIG(self.arms, device)
        self.histogram = torch.zeros(self.arms, device=device)
        self.n = 0
        self.ngpu = torch.zeros(1, device=device)
        self.device = device

    def step(self):
        samples, _ = self.belief.sample()
        play = torch.argmax(samples)
        obs = self.stepper(play)
        self.belief.update_component(play, obs)
        self.histogram[play] += 1
        self.n += 1
        self.ngpu += 1
        if self.arms == 2:
            pass # Special analytic case
        if self.n >= 10:
            #print(torch.max(self.histogram) / self.ngpu)
            if (torch.max(self.histogram) / self.ngpu) > 0.80:
                choice = torch.argmax(self.histogram)
                self.stepper.choose(choice)
                self.ngpu.zero_()
                self.n = 0
                self.histogram.zero_()

                    
                    
                self.belief.m = self.belief.m[choice].repeat(self.arms)
                self.belief.vinv = self.belief.vinv[choice].repeat(self.arms)
                self.belief.a = self.belief.a[choice].repeat(self.arms)
                self.belief.b = self.belief.b[choice].repeat(self.arms)
                self.belief.vinv /= 2
                self.belief.a /= 2
                self.belief.b /= 2
        return obs
