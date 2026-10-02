"""Testes do pipeline v2. Rodar na raiz do projeto:

    .venv\\Scripts\\python.exe -m unittest -v tests.test_pipeline_v2
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import pipeline_v2 as pv  # noqa: E402

DATA_DIR = RAIZ / 'data' / 'raw'


# ── Código LEGADO do NB08 (células [6], [7] e [11]), copiado só para os testes de regressão ──
def _nb08_build_features(series, n_lags=12, season=12):
    df = pd.DataFrame({'y': series})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['roll_mean_3'] = df['y'].shift(1).rolling(3).mean()
    df['roll_std_3'] = df['y'].shift(1).rolling(3).std()
    df['roll_mean_6'] = df['y'].shift(1).rolling(6).mean()
    df['lag_season'] = df['y'].shift(season)
    df['lag_season2'] = df['y'].shift(season * 2)
    df.dropna(inplace=True)
    return df.drop(columns='y').values, df['y'].values


def _nb08_mape_safe(y_true, y_pred):
    ok = np.abs(y_true) > 1e-8
    if ok.sum() < len(y_true) * 0.5:
        den = (np.abs(y_true) + np.abs(y_pred)) / 2
        m = den > 1e-8
        return np.mean(np.abs(y_true[m] - y_pred[m]) / den[m]) * 100
    return np.mean(np.abs((y_true[ok] - y_pred[ok]) / y_true[ok])) * 100


def _nb08_janelas(n, h=18, n_origins=3, min_train=24):
    out = []
    for off in range(n_origins, 0, -1):
        fim = n - (off - 1) * h
        ini = fim - h
        if ini >= min_train and fim <= n:
            out.append((ini, fim))
    return out


def _nb08_perturb_outliers(series, outlier_frac, rng):
    s = series.copy()
    sigma = np.std(s)
    n_out = max(1, int(len(s) * outlier_frac))
    idx = rng.choice(len(s), size=n_out, replace=False)
    signs = rng.choice([-1, 1], size=n_out)
    s[idx] = np.mean(s) + signs * 3 * sigma
    return s


class TestC1RecursaoSemDeslocamento(unittest.TestCase):
    """C1: série y = 100 + t com regressão linear perfeita; o correto é prever 180..184."""

    def setUp(self):
        self.y = 100 + np.arange(80, dtype=float)
        self.real = np.array([180., 181., 182., 183., 184.])

    def test_v2_preve_180_a_184(self):
        pred, status = pv.ml_forecast(self.y, 5, LinearRegression())
        print(f'\n    v2: {np.round(pred, 4)}')
        self.assertEqual(status, pv.OK)
        np.testing.assert_allclose(pred, self.real, atol=1e-6)

    def test_v2_svr_com_escala_tambem_usa_o_proximo_passo(self):
        pred, _ = pv.ml_forecast(self.y, 5, LinearRegression(), escalar=True)
        np.testing.assert_allclose(pred, self.real, atol=1e-6)

    def test_legado_nb08_desloca_um_passo(self):
        """Documenta o defeito: o laço do NB08 prevê 179 (o último valor observado)."""
        X, yy = _nb08_build_features(self.y)
        lr = LinearRegression().fit(X, yy)
        hist, preds = list(self.y), []
        for _ in range(5):
            Xt, _ = _nb08_build_features(np.array(hist))
            preds.append(lr.predict(Xt[[-1]])[0])
            hist.append(preds[-1])
        print(f'\n    NB08 (legado): {np.round(preds, 2)}')
        self.assertAlmostEqual(preds[0], 179.0, places=6)


class TestC2PerturbacaoSoNoTreino(unittest.TestCase):
    """C2: outliers não podem contaminar o teste."""

    @classmethod
    def setUpClass(cls):
        cls.ids, cls.series = pv.load_m4_piloto(DATA_DIR, n=200)

    def test_legado_nb08_numeros_documentados(self):
        """Reproduz a tabela da Fase 0: limpo 9,99; NB08 11,63; só treino 10,08 (Persistência, outliers 2%)."""
        rng = np.random.default_rng(42)
        pert = [_nb08_perturb_outliers(s, 0.02, rng) for s in self.series]

        def med(fn):
            return np.nanmedian([np.median(v) if (v := fn(i)) else np.nan for i in range(len(self.series))])
        limpo = med(lambda i: [_nb08_mape_safe(self.series[i][a:b], np.full(b - a, self.series[i][a - 1]))
                               for a, b in _nb08_janelas(len(self.series[i]))])
        nb08 = med(lambda i: [_nb08_mape_safe(pert[i][a:b], np.full(b - a, pert[i][a - 1]))
                              for a, b in _nb08_janelas(len(self.series[i]))])
        treino = med(lambda i: [_nb08_mape_safe(self.series[i][a:b], np.full(b - a, pert[i][a - 1]))
                                for a, b in _nb08_janelas(len(self.series[i]))])
        print(f'\n    legado: limpo={limpo:.2f}  NB08={nb08:.2f} ({100*(nb08-limpo)/limpo:+.1f}%)  '
              f'só treino={treino:.2f} ({100*(treino-limpo)/limpo:+.1f}%)')
        self.assertAlmostEqual(limpo, 9.99, places=2)
        self.assertAlmostEqual(nb08, 11.63, places=2)
        self.assertAlmostEqual(treino, 10.08, places=2)

    def test_v2_teste_permanece_limpo(self):
        """Um modelo que prevê zero tem MAE = média |teste limpo|, mesmo com outliers de 5%."""
        def zero(train, h):
            return np.zeros(h), pv.OK
        s = self.series[0]
        regs = pv.avaliar_serie(s, 'x', 0, {'zero': zero}, 18, cenario='C_outliers_5pct')
        for r in regs:
            o = r['origin']
            self.assertAlmostEqual(r['mae'], np.mean(np.abs(s[o:o + 18])), places=8)

    def test_v2_persistencia_outliers_m4(self):
        """No v2 (teste limpo) o efeito dos outliers na Persistência é pequeno; o alvo ruidoso o infla."""
        mods = ['persistence']
        base = pv.agregar_por_serie(pv.avaliar_base(self.series, self.ids, mods, 18, n_jobs=1))
        treino = pv.agregar_por_serie(pv.avaliar_base(self.series, self.ids, mods, 18, 'C_outliers_2pct', n_jobs=1))
        ruid = pv.agregar_por_serie(pv.avaliar_base(self.series, self.ids, mods, 18, 'C_outliers_2pct',
                                                    alvo_ruidoso=True, n_jobs=1))
        d_t = pv.tabela_delta(base, treino).loc['persistence', 'delta_pct']
        d_r = pv.tabela_delta(base, ruid).loc['persistence', 'delta_pct']
        print(f'\n    v2 Persistência, outliers 2%: ΔMAPE só treino={d_t:+.1f}%  alvo ruidoso={d_r:+.1f}%')
        self.assertLess(abs(d_t), 5.0)
        self.assertGreater(d_r, d_t + 5.0)


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.s = np.random.default_rng(0).normal(100, 10, 150).cumsum()

    def test_sem_colunas_duplicadas(self):
        for lag24 in (False, True):
            X, _, nomes = pv.build_features(self.s, com_lag24=lag24)
            self.assertEqual(len(nomes), len(set(nomes)))
            self.assertNotIn('lag_season', nomes)
            self.assertEqual(len(nomes), 16 if lag24 else 15)
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    self.assertFalse(np.allclose(X[:, i], X[:, j]), f'{nomes[i]} == {nomes[j]}')

    def test_semantica_dos_lags(self):
        X, y, nomes = pv.build_features(self.s, com_lag24=True)
        t = 24 + 7   # linha de alvo s[t]
        linha = X[t - 24]
        self.assertEqual(y[t - 24], self.s[t])
        self.assertEqual(linha[nomes.index('lag_1')], self.s[t - 1])
        self.assertEqual(linha[nomes.index('lag_12')], self.s[t - 12])
        self.assertEqual(linha[nomes.index('lag_24')], self.s[t - 24])
        self.assertAlmostEqual(linha[nomes.index('roll_mean_6')], self.s[t - 6:t].mean())

    def test_proximo_passo_igual_a_linha_de_treino(self):
        """A linha usada na previsão de s[t] é idêntica à linha de treino com alvo s[t]."""
        X, _, _ = pv.build_features(self.s)
        m = pv.historico_minimo()
        for t in [m, 40, 99, len(self.s) - 1]:
            np.testing.assert_array_equal(pv.next_step_features(self.s[:t])[0], X[t - m])

    def test_primeira_janela_tem_linhas(self):
        X, _, _ = pv.build_features(self.s[:24])
        self.assertEqual(len(X), 12)


class TestLimeCasamentoPorIndice(unittest.TestCase):
    """O alvo depende só de lag_10 (e depois só de lag_12): o LIME deve apontá-los."""

    def _entry(self, coluna):
        s = np.random.default_rng(1).normal(0, 1, 300)
        X, _, nomes = pv.build_features(s)
        y = 5 * X[:, nomes.index(coluna)]
        return {'model': LinearRegression().fit(X, y), 'X': X, 'sx': None, 'sy': None}, nomes

    def test_lag_10(self):
        e, nomes = self._entry('lag_10')
        imp = pv.lime_importancias(e, nomes)
        self.assertEqual(nomes[int(np.argmax(imp))], 'lag_10')
        # Com 200 amostras o 2º colocado chega a ~70% de lag_10 (ruído de amostragem do LIME);
        # com 5000 amostras lag_10 domina com folga.
        imp5k = pv.lime_importancias(e, nomes, num_samples=5000)
        self.assertGreater(imp5k[nomes.index('lag_10')], 2 * np.sort(imp5k)[-2])

    def test_legado_substring_perde_lag_10(self):
        """Mesma explicação, associação por substring do NB09 original: lag_10 vira zero e cai em lag_1."""
        from lime.lime_tabular import LimeTabularExplainer
        e, nomes = self._entry('lag_10')
        exp = LimeTabularExplainer(e['X'], feature_names=nomes, mode='regression', random_state=pv.SEED,
                                   verbose=False).explain_instance(e['X'][-1], e['model'].predict,
                                                                   num_features=len(nomes), num_samples=200)
        legado = np.zeros(len(nomes))
        for fname, val in dict(exp.as_list()).items():
            for j, fn in enumerate(nomes):
                if fn in fname:
                    legado[j] += abs(val)
                    break
        print(f"\n    lag_10: índice={pv.lime_importancias(e, nomes)[nomes.index('lag_10')]:.3f}  "
              f"substring={legado[nomes.index('lag_10')]:.3f} | lag_1 por substring={legado[nomes.index('lag_1')]:.3f}")
        self.assertEqual(legado[nomes.index('lag_10')], 0.0)
        self.assertEqual(nomes[int(np.argmax(legado))], 'lag_1')

    def test_lag_12(self):
        e, nomes = self._entry('lag_12')
        imp = pv.lime_importancias(e, nomes)
        self.assertEqual(nomes[int(np.argmax(imp))], 'lag_12')


class TestShapSvr(unittest.TestCase):
    def test_reprodutivel_e_ordem_por_indice(self):
        s = 100 + 10 * np.sin(np.arange(200) * 2 * np.pi / 12) + np.random.default_rng(3).normal(0, 1, 200)
        fit = pv.treinar_modelos_xai(s)
        a = pv.shap_importancias(fit['SVR'], 'SVR')
        b = pv.shap_importancias(fit['SVR'], 'SVR')
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), len(fit['nomes']))


class TestJanelasEPerturbacoes(unittest.TestCase):
    def test_janelas(self):
        for n, h in [(51, 18), (66, 18), (330, 24), (918, 18)]:
            origens = pv.make_rolling_windows(n, h)
            self.assertEqual(len(origens), 3)
            self.assertEqual(origens[0], pv.MIN_TRAIN)
            self.assertEqual(origens[-1] + h, n)
        self.assertEqual(pv.make_rolling_windows(41, 18), [])

    def test_modelo_so_recebe_o_treino(self):
        vistos = []

        def espiao(train, h):
            vistos.append(train.copy())
            return np.zeros(h), pv.OK
        s = np.arange(100, dtype=float)
        pv.avaliar_serie(s, 'x', 0, {'e': espiao}, 18)
        for o, tr in zip(pv.make_rolling_windows(100, 18), vistos):
            np.testing.assert_array_equal(tr, s[:o])

    def test_semente_por_janela(self):
        tr = np.arange(60, dtype=float)
        a = pv.perturbar(tr, 'A_ruido_30pct', pv.rng_janela(5, 24, 'A_ruido_30pct'))
        b = pv.perturbar(tr, 'A_ruido_30pct', pv.rng_janela(5, 24, 'A_ruido_30pct'))
        c = pv.perturbar(tr, 'A_ruido_30pct', pv.rng_janela(5, 36, 'A_ruido_30pct'))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(a, c))

    def test_truncamento_mantem_o_mais_recente(self):
        tr = np.arange(100, dtype=float)
        np.testing.assert_array_equal(pv.perturbar(tr, 'D_trunca_25pct', None), tr[-25:])
        np.testing.assert_array_equal(pv.perturbar(tr[:60], 'D_trunca_25pct', None), tr[36:60])

    def test_ausencias_sem_nan(self):
        tr = np.arange(80, dtype=float)
        p = pv.perturbar(tr, 'B_ausencias_30pct', pv.rng_janela(0, 80, 'B_ausencias_30pct'))
        self.assertFalse(np.isnan(p).any())
        np.testing.assert_allclose(p[5:-5], tr[5:-5], atol=1e-9)   # no interior, a interpolação reproduz a reta
        self.assertEqual(len(p), len(tr))


class TestMetricasEFallbacks(unittest.TestCase):
    def test_mape_e_r2(self):
        m = pv.calc_metrics(np.array([0., 10., 20.]), np.array([1., 11., 18.]))
        self.assertAlmostEqual(m['mape'], 10.0)
        self.assertTrue(np.isnan(pv.calc_metrics(np.zeros(5), np.ones(5))['r2']))

    def test_fallbacks_registrados(self):
        self.assertEqual(pv.model_holt_winters(np.arange(20, dtype=float), 6)[1], pv.FB_SNAIVE)
        self.assertEqual(pv.model_sarima_heuristic(np.zeros(40), 6)[1], pv.FB_SNAIVE)
        regs = pv.avaliar_serie(np.zeros(60), 'z', 0, ['sarima_heuristic', 'svr'], 18)
        self.assertEqual({r['status'] for r in regs if r['model'] == 'sarima_heuristic'}, {pv.FB_SNAIVE})
        self.assertEqual({r['status'] for r in regs if r['model'] == 'svr'}, {pv.FB_MEAN})

    def test_previsao_invalida_vira_fallback(self):
        regs = pv.avaliar_serie(np.arange(60, dtype=float), 'x', 0,
                                {'nan': lambda tr, h: (np.full(h, np.nan), pv.OK)}, 18)
        self.assertEqual({r['status'] for r in regs}, {pv.FB_SNAIVE})


class TestSarimaDegenerado(unittest.TestCase):
    """Com 24 obs e diferença sazonal a verossimilhança é calculada sobre zero observações (AIC = 2k)."""

    def setUp(self):
        r = np.random.default_rng(7)
        self.s = 100 + 10 * np.sin(np.arange(120) * 2 * np.pi / 12) + r.normal(0, 2, 120)

    def test_heuristico_24_obs_vira_fallback_registrado(self):
        pred, status, info = pv.model_sarima_heuristic(self.s[:24], 18)
        self.assertEqual(status, pv.FB_SNAIVE)
        self.assertEqual(info['motivo'], 'verossimilhanca_degenerada')

    def test_selecao_nunca_escolhe_degenerado(self):
        for n in (24, 51, 100):
            escolha, info = pv.selecionar_ordem_sarima(self.s[:n])
            if escolha is not None:
                fit = pv._sarimax_fit(self.s[:n], *escolha)
                self.assertFalse(pv.verossimilhanca_degenerada(fit))
                self.assertGreater(info['aic'], 2 * len(fit.params))   # AIC = 2k seria o sinal de degeneração

    def test_heuristico_serie_longa_ok(self):
        self.assertEqual(pv.model_sarima_heuristic(self.s, 18)[1], pv.OK)


class TestExplosivas(unittest.TestCase):
    def test_agregacao_com_e_sem_explosivas(self):
        df = pd.DataFrame({'model': 'm', 'unique_id': 'u', 'status': pv.OK, 'janela': [1, 2, 3],
                           'mape': [10.0, 1e20, 30.0], 'mae': 1.0, 'r2': 0.0, 'exec_time_s': 0.0,
                           'info': ['', json.dumps({'explosiva': True}), '']})
        self.assertEqual(pv.flag_explosiva(df).tolist(), [False, True, False])
        self.assertEqual(pv.agregar_por_serie(df)['mape'].iloc[0], 30.0)
        self.assertEqual(pv.agregar_por_serie(df, excluir_explosivas=True)['mape'].iloc[0], 20.0)


class TestXaiMesmosPontos(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(11)
        self.s = 100 + 10 * np.sin(np.arange(160) * 2 * np.pi / 12) + r.normal(0, 2, 160)

    def test_shap_e_lime_explicam_os_mesmos_pontos(self):
        d = pd.DataFrame(pv.explicar_serie(self.s, 'x', 0, 18, 'limpo'))
        for m in pv.MODELOS_XAI:
            sh = d[(d.modelo == m) & (d.metodo == 'shap')][['ponto', 'tipo_ponto']].reset_index(drop=True)
            li = d[(d.modelo == m) & (d.metodo == 'lime')][['ponto', 'tipo_ponto']].reset_index(drop=True)
            pd.testing.assert_frame_equal(sh, li)
            self.assertEqual(len(sh), pv.N_PONTOS_XAI)
            self.assertEqual(sh['tipo_ponto'].iloc[-1], 'previsao')

    def test_ultimo_ponto_e_o_de_previsao(self):
        train = self.s[:-18]
        pts, tipos = pv.pontos_explicados(train)
        np.testing.assert_array_equal(pts[-1], pv.next_step_features(train)[0])
        X, _, _ = pv.build_features(train)
        np.testing.assert_array_equal(pts[:-1], X[-(pv.N_PONTOS_XAI - 1):])

    def test_ruido_depende_so_do_treino(self):
        outra = self.s.copy()
        outra[-18:] += 1e6   # muda só o teste
        np.testing.assert_array_equal(pv.treino_xai(self.s, 18, 'ruido30', 3), pv.treino_xai(outra, 18, 'ruido30', 3))
        self.assertFalse(np.allclose(pv.treino_xai(self.s, 18, 'ruido30', 3), self.s[:-18]))


class TestBootstrap(unittest.TestCase):
    def test_ic_deterministico_e_contem_estimativa(self):
        v = np.random.default_rng(0).normal(5, 1, 50)
        a, b = pv.bootstrap_ic(v), pv.bootstrap_ic(v)
        self.assertEqual(a, b)
        self.assertLess(a[1], a[0]); self.assertLess(a[0], a[2])

    def test_delta_pareado(self):
        r = np.random.default_rng(1)
        ids = [f's{i}' for i in range(60)]
        base = pd.DataFrame([{'model': m, 'unique_id': u, 'mape': 10 + r.normal()} for m in ['a', 'b'] for u in ids])
        adv = base.copy()
        adv.loc[adv.model == 'a', 'mape'] += 5     # 'a' degrada; 'b' não muda
        tab, pares = pv.bootstrap_delta(base, adv, ['a', 'b'], n=500)
        self.assertFalse(tab.loc['a', 'ic_inclui_zero'])
        self.assertTrue(tab.loc['b', 'ic_inclui_zero'])
        self.assertFalse(pares['ic_inclui_zero'].iloc[0])


class TestBootstrapRanks(unittest.TestCase):
    def test_ranks_e_ic(self):
        r = np.random.default_rng(2)
        ids = [f's{i}' for i in range(80)]
        bs = pd.DataFrame([{'model': m, 'unique_id': u, 'mae': base + r.normal(0, 0.1)}
                           for m, base in [('bom', 1.0), ('medio', 2.0), ('ruim', 3.0)] for u in ids])
        tab, boot = pv.bootstrap_ranks(bs, ['ruim', 'bom', 'medio'], metrica='mae', n=300)
        self.assertEqual(tab['rank'].to_dict(), {'ruim': 3, 'bom': 1, 'medio': 2})
        self.assertEqual(tab.loc['bom', 'rank_ic_sup'], 1)
        self.assertEqual(boot.shape, (300, 3))


class TestMase(unittest.TestCase):
    def test_escala_e_mase(self):
        s = np.r_[np.arange(12.0), np.arange(12.0) + 2, np.arange(12.0) + 4]   # diferença sazonal = 2
        self.assertAlmostEqual(pv.escala_mase(s[:24]), 2.0)
        self.assertTrue(np.isnan(pv.escala_mase(np.tile(np.arange(12.0), 2))))   # sazonalidade perfeita: escala 0
        self.assertTrue(np.isnan(pv.escala_mase(np.zeros(30))))
        self.assertTrue(np.isnan(pv.escala_mase(np.arange(12.0))))
        df = pd.DataFrame({'unique_id': ['a', 'a'], 'origin': [24, 30], 'mae': [4.0, 1.0], 'model': 'm',
                           'status': pv.OK, 'mape': 1.0, 'r2': 0.0, 'exec_time_s': 0.0})
        out = pv.adicionar_mase(df, {'a': s})
        self.assertEqual(out['mase'].tolist(), [2.0, 0.5])
        self.assertIn('mase', pv.agregar_por_serie(out).columns)


class TestMeta(unittest.TestCase):
    def test_meta_json(self):
        with tempfile.TemporaryDirectory() as d:
            dado = Path(d) / 'dado.txt'
            dado.write_text('abc', encoding='utf-8')
            df = pv.cache_csv(Path(d) / 'x.csv', lambda: pd.DataFrame({'a': [1]}), {'etapa': 'teste'},
                              arquivos_dados=[dado])
            meta = json.loads((Path(d) / 'x.csv.meta.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['pipeline_sha256'], pv.hash_pipeline())
            self.assertEqual(meta['sementes']['SEED'], 42)
            self.assertIn('scikit-learn', meta['versoes'])
            self.assertEqual(meta['etapa'], 'teste')
            self.assertEqual(meta['dados'][0]['sha256'],
                             'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')   # SHA-256('abc')
            self.assertEqual(len(df), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
