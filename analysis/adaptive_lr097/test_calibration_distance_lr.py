import copy, unittest
from calibration_distance_lr import CalibrationDistanceLR, Config

class ControllerTests(unittest.TestCase):
    def observe(self, c, values):
        return [c.observe(i+1, v, split_role='calibration') for i,v in enumerate(values)]
    def test_improving_distance_keeps_rates(self):
        c=CalibrationDistanceLR(Config(interval=1,warmup=2))
        rows=self.observe(c,[1/(i+1) for i in range(20)])
        self.assertTrue(all(r['lr_scale']==1 for r in rows))
    def test_plateau_patience_cooldown_floor(self):
        c=CalibrationDistanceLR(Config(interval=1,warmup=2))
        rows=self.observe(c,[1.]*20)
        self.assertEqual([r['completed_updates'] for r in rows if r['action']=='reduce'],[6,10,14])
        self.assertEqual(c.scale,.125)
    def test_one_spike_does_not_reduce(self):
        c=CalibrationDistanceLR(Config(interval=1,warmup=2))
        self.observe(c,[.5,.4,.3,99.,.29,.28,.27])
        self.assertEqual(c.scale,1.)
    def test_invalid_or_noncalibration_input_preserves_state(self):
        c=CalibrationDistanceLR(Config(interval=1,warmup=0));c.observe(1,1.,split_role='calibration')
        for step,value,role in [(2,.9,'evaluation'),(2,float('nan'),'calibration'),(2,-1.,'calibration'),(1,.9,'calibration'),(3,.9,'calibration')]:
            before=c.state_dict()
            with self.assertRaises(ValueError):c.observe(step,value,split_role=role)
            self.assertEqual(c.state_dict(),before)
    def test_checkpoint_replay_and_atomic_corruption_rejection(self):
        cfg=Config(interval=1,warmup=2);a=CalibrationDistanceLR(cfg)
        self.observe(a,[1.]*8);b=CalibrationDistanceLR(cfg);b.load_state_dict(a.state_dict())
        for step in range(9,20):
            self.assertEqual(a.observe(step,1.,split_role='calibration'),b.observe(step,1.,split_role='calibration'))
        bad=copy.deepcopy(b.state_dict());bad['decisions'][-1]['lr_scale']=.75;before=b.state_dict()
        with self.assertRaises(ValueError):b.load_state_dict(bad)
        self.assertEqual(before,b.state_dict())
    def test_native_pg_callback_all_three_rates_and_exact_resume(self):
        import torch
        from particlegan import GANTrainer,get_recipe
        from state_fingerprint import fingerprint
        torch.set_num_threads(1)
        rec=get_recipe('gan',trainer='bispf_optimized',total_steps=20000,batch_size=32,
                       num_particles=512,z_dim=8,lr=.012,d_lr_mult=1.5,prior_lr_mult=5.)
        cfg=Config(interval=1,warmup=0,window=1,patience=1,cooldown=0)
        real=torch.linspace(-1,1,32*150).reshape(32,150)
        def build():
            torch.manual_seed(19)
            g=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.Tanh(),torch.nn.Linear(16,150))
            d=torch.nn.Sequential(torch.nn.Linear(150,16),torch.nn.LeakyReLU(.2),torch.nn.Linear(16,1))
            t=GANTrainer(rec,g,d,prior=rec.make_prior(),seed=17,max_steps=20000,
                         serve_after_step=False,penalty_options={'input_domain':'continuous'})
            c=CalibrationDistanceLR(cfg);t.policy.schedule=c
            self.assertIsNone(t.policy.controller);self.assertIsNone(t.policy.lr_settle)
            self.assertEqual(rec.ema_decay,0)
            return t,c
        def step(t,c):
            prior_scale=c.scale;t.step(real)
            rates=[t.opt_g.param_groups[0]['lr'],t.opt_g.param_groups[1]['lr'],t.opt_d.param_groups[0]['lr']]
            for actual,base in zip(rates,[.012,.06,.018]):self.assertAlmostEqual(actual,base*prior_scale,places=14)
            before=fingerprint(t.state_dict());c.observe(t.completed_steps,1.,split_role='calibration')
            self.assertEqual(before,fingerprint(t.state_dict()))
        t,c=build()
        for _ in range(3):step(t,c)
        saved=copy.deepcopy(t.state_dict());saved_control=c.state_dict()
        for _ in range(3):step(t,c)
        expected=fingerprint(t.state_dict());expected_control=c.state_dict()
        other,other_control=build();other_control.load_state_dict(saved_control);other.load_state_dict(saved)
        for _ in range(3):step(other,other_control)
        self.assertEqual(expected,fingerprint(other.state_dict()))
        self.assertEqual(expected_control,other_control.state_dict())
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main(verbosity=2)
