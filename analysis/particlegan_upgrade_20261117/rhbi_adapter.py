"""Declared conditional RHBI variant, derived from frozen ALZ caller loop.

Only unsupported unconditional simplex charts are disabled via native recipe
field rhbi_simplex_weight=0. The variant keeps all other optimized defaults.
Native record binding, sampled rows, conditional transport, step damping,
penalty, prior regularization and update policy run in the released library.
"""
NAME = 'rhbi_optimized_alz_conditional_no_simplex'


def derive(body):
    from host import replace_once
    start = body.index('def recipe(topt=None):')
    end = body.index('def expected_lrs(', start)
    replacement = '''def recipe(topt=None):
    if topt is not None and not topt.is_default:
        raise ValueError('release pilot permits only the fixed pbulk host')
    return get_recipe('gan', trainer='rhbi_optimized', total_steps=20000,
                      batch_size=32, num_particles=512, z_dim=8,
                      rhbi_simplex_weight=0.).replace(name='rhbi_optimized_alz_conditional_no_simplex')


def check_recipe(rec, topt):
    if rec.to_dict() != recipe(topt).to_dict():
        raise ValueError('declared conditional RHBI variant differs')
    return rec


'''
    body = body[:start] + replacement + body[end:]
    body = replace_once(body, "T005 = {'generator_lr': .012, 'critic_lr': .018, 'particle_lr': .06, 'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .01}",
        "T005 = {'generator_lr': .014, 'critic_lr': .021, 'particle_lr': .0406, 'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .02}")
    body = replace_once(body, 'from particlegan.bispf import bind_thermal_source\n    bind_thermal_source(od.record, loss, reg)',
        'from particlegan.rhbi import bind_rhbi_record\n    bind_rhbi_record(od.record, loss, reg)')
    body = body.replace("'native_recipe': 'bispf_optimized'", f"'native_recipe': {NAME!r}")
    body = replace_once(body, "q.get('native_recipe') != 'bispf_optimized'", f"q.get('native_recipe') != {NAME!r}")
    body = replace_once(body, 'noise = policy.begin_step(real, execution_limit=execution_limit)',
        'noise = policy.begin_step(real, game_record=pen.regularizer.record, execution_limit=execution_limit)')
    body = replace_once(body, 'f = fake(z, c2, lib2); lg = loss.g_loss(D(f, c2))',
        '''f = fake(z, c2, lib2)
        og.bind_origin_sample(f)
        loss.rhbi.bind_discrete_prior_step(True, simplicial_active=False)
        lg = loss.g_loss(D(f, c2), D(real2, c2))''')
    # Match the native caller's finally reset before finish_step, even on errors.
    start = body.index('        real2, c2, lib2 = ctx.batch', body.index('def advance('))
    stop = body.index('        usage.observe(', start)
    phase = body[start:stop]
    phase = replace_once(phase,
        '        policy.after_generator_step(); D.requires_grad_(True); policy.finish_step()\n',
        '        policy.after_generator_step()\n')
    phase = '        try:\n' + ''.join('    ' + line if line.strip() else line for line in phase.splitlines(True))
    phase += '''        finally:
            loss.rhbi.bind_discrete_prior_step(False)
            D.requires_grad_(True)
        policy.finish_step()
'''
    body = body[:start] + phase + body[stop:]
    body = replace_once(body, "         'wall_seconds': wall_seconds}",
        "         'wall_seconds': wall_seconds, 'rhbi_loss_state': loss.rhbi.state_dict()}")
    body = replace_once(body,
        "policy.load_state_dict(q['policy']); rec.restore_prior_regularizer(b[-2], q['prior_regularizer'].get('prior_regularizer'))",
        "policy.load_state_dict(q['policy']); rec.restore_prior_regularizer(b[-2], q['prior_regularizer'].get('prior_regularizer'))\n    b[7].rhbi.load_state_dict(q['rhbi_loss_state'])")
    return body
