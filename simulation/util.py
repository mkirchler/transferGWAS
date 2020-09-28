from os.path import join

LATENT_DIR = 'latent'

IMG_DIR = 'images'
EMB_DIR = 'embeddings'
OUT_DIR = 'results'

PATH_TO_EXPORT = '../feature_condensation/feature_condensation.py'
PATH_TO_GWAS = '../lmm/run_lmm.py'


def get_latent_pkl(exp_var, n_causal, seed):
    latent_pkl_templ = 'exp_var%.3f_nc%d_seed%d.pkl'
    return latent_pkl_templ % (exp_var, n_causal, seed)


def get_latent_bolt(exp_var, n_causal, seed):
    latent_bolt_templ = 'exp_var%.3f_nc%d_seed%d.txt'
    return latent_bolt_templ % (exp_var, n_causal, seed)


def get_out(
        gan_name,
        exp_var,
        n_causal,
        mult_scale,
        model_name,
        layer,
        spatial,
        img_size,
        tfms,
        seed,
        sample_size,
        ):
    out_templ = 'exp_%s_var%.3f_nc%d_sc%.2f_%s_%s_%s_s%d_%s_seed%d_ss%d_pheno_%s'
    return out_templ % (
            gan_name,
            exp_var,
            n_causal,
            mult_scale,
            model_name,
            layer,
            spatial,
            img_size,
            tfms,
            seed,
            sample_size,
            '%s',           # feature dimension
            )


def get_img_dir(gan_name, exp_var, n_causal, mult_scale, seed):
    img_dir_templ = 'exp_%s_var%.3f_nc%d_sc%.2f_seed%d'
    return join(
            IMG_DIR,
            img_dir_templ % (gan_name, exp_var, n_causal, mult_scale, seed),
            )


def get_emb(
        gan_name,
        exp_var,
        n_causal,
        mult_scale,
        model_name,
        layer,
        spatial,
        img_size,
        tfms,
        seed,
        options,
        ):
    emb_templ = 'exp_%s_var%.3f_nc%d_sc%.2f_%s_%s_%s_s%d_%s_seed%d%s.txt'
    return emb_templ % (
            gan_name,
            exp_var,
            n_causal,
            mult_scale,
            model_name,
            layer,
            spatial,
            img_size,
            tfms,
            seed,
            options,
            )
