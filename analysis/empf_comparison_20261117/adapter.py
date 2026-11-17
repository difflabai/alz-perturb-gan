"""Task adapter for native EMPF and one explicitly matched public variant."""
NAMES = {'native': 'empf', 'matched': 'empf_alz_rhbi_shared_hyperparameters'}
MATCHED = dict(training_precision='float32', lr=.014, prior_lr_mult=2.9,
               ebi_beta=12., prior_reg=.02, optimizer_smoothing=.0061,
               optimizer_svd_backend='native', optimizer_dense_profile='polar_cpu_v1')


def derive(body, lane):
    from host import replace_once
    start = body.index('def recipe(topt=None):')
    stop = body.index('def expected_lrs(', start)
    overrides = MATCHED if lane == 'matched' else {}
    replacement = f'''def recipe(topt=None):
    if topt is not None and not topt.is_default:
        raise ValueError('EMPF pilot permits only the fixed pbulk host')
    return get_recipe('gan', trainer='empf', total_steps=20000,
                      batch_size=32, num_particles=512, z_dim=8, reg_every=1,
                      **{overrides!r}).replace(name={NAMES[lane]!r})


def check_recipe(rec, topt):
    if rec.to_dict() != recipe(topt).to_dict():
        raise ValueError('declared EMPF recipe differs')
    return rec


'''
    body = body[:start] + replacement + body[stop:]
    values = dict(generator_lr=.014, critic_lr=.021, particle_lr=.0406,
                  reg_coeff=1., reg_kappa=1., prior_reg=.02) if lane == 'matched' else dict(
                  generator_lr=.012, critic_lr=.018, particle_lr=.03,
                  reg_coeff=1., reg_kappa=1., prior_reg=.01)
    body = replace_once(body,
        "T005 = {'generator_lr': .012, 'critic_lr': .018, 'particle_lr': .06, 'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .01}",
        f'T005 = {values!r}')
    body = replace_once(body,
        'from particlegan.bispf import bind_thermal_source\n    bind_thermal_source(od.record, loss, reg)',
        'from particlegan.empf import bind_empf_record\n    bind_empf_record(od.record, loss, reg)')
    body = body.replace("'native_recipe': 'bispf_optimized'", f"'native_recipe': {NAMES[lane]!r}")
    body = replace_once(body, "q.get('native_recipe') != 'bispf_optimized'",
                        f"q.get('native_recipe') != {NAMES[lane]!r}")
    body = replace_once(body, 'noise = policy.begin_step(real, execution_limit=execution_limit)',
        'noise = policy.begin_step(real, game_record=pen.regularizer.record, execution_limit=execution_limit)')
    return body
