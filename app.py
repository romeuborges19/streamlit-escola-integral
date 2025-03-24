import matplotlib
from matplotlib import gridspec
import matplotlib.colors
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st

st.title("Análise de Dados - Programa Escola Integral")


def get_integral(df):
    df["IN_INT"] = (
        df["QT_MAT_FUND_AI_INT"] + df["QT_MAT_FUND_AF_INT"] + df["QT_MAT_MED_INT"]
    ) > 0
    return df


@st.cache_data
def get_data(mod: bool):
    df = pd.read_parquet("censo_rend_2.parquet")
    df = get_integral(df)

    geo_df = gpd.read_file(r"BR_Municipios_2023.shp")

    groupby_fields = ["ANO", "NO_REGIAO", "SG_UF", "CO_MUNICIPIO", "TP_DEPENDENCIA"]

    if mod is not None:
        groupby_fields.append("IN_INT")

    df = (
        df.groupby(by=groupby_fields)[
            [
                "APROV_FUND",
                "APROV_FUND1",
                "APROV_FUND2",
                "APROV_MED",
                "REPROV_FUND",
                "REPROV_FUND1",
                "REPROV_FUND2",
                "REPROV_MED",
                "ABAND_FUND",
                "ABAND_FUND1",
                "ABAND_FUND2",
                "ABAND_MED",
            ]
        ]
        .mean()
        .reset_index()
    )

    if mod is not None:
        df = df[df.IN_INT == mod]

    return df, geo_df


@st.cache_data
def get_data_matriculas(mod: bool | None):
    df = pd.read_parquet("censo_rend_2.parquet")
    geo_df = gpd.read_file(r"BR_Municipios_2023.shp")

    df = get_integral(df)

    df_mat = df.rename(
        columns={
            "QT_MAT_INF_PRE": "PARCIAL_PRE",
            "QT_MAT_INF_CRE": "PARCIAL_CRE",
            "QT_MAT_FUND_AI": "PARCIAL_FUND1",
            "QT_MAT_FUND_AF": "PARCIAL_FUND2",
            "QT_MAT_MED": "PARCIAL_MED",
            "QT_MAT_INF_PRE_INT": "INTEGRAL_PRE",
            "QT_MAT_INF_CRE_INT": "INTEGRAL_CRE",
            "QT_MAT_FUND_AI_INT": "INTEGRAL_FUND1",
            "QT_MAT_FUND_AF_INT": "INTEGRAL_FUND2",
            "QT_MAT_MED_INT": "INTEGRAL_MED",
        }
    )

    groupby_fields = ["ANO", "NO_REGIAO", "SG_UF", "CO_MUNICIPIO", "TP_DEPENDENCIA"]

    if mod is not None:
        groupby_fields.append("IN_INT")

    df_mat = (
        df_mat.groupby(by=groupby_fields)[
            [
                "INTEGRAL_PRE",
                "INTEGRAL_CRE",
                "INTEGRAL_FUND1",
                "INTEGRAL_FUND2",
                "INTEGRAL_MED",
                "PARCIAL_PRE",
                "PARCIAL_CRE",
                "PARCIAL_FUND1",
                "PARCIAL_FUND2",
                "PARCIAL_MED",
            ]
        ]
        .sum()
        .reset_index()
    )

    df_mat[f"INTEGRAL_FUND"] = df_mat[f"INTEGRAL_FUND1"] + df_mat[f"INTEGRAL_FUND2"]
    df_mat[f"PARCIAL_FUND"] = df_mat[f"PARCIAL_FUND1"] + df_mat[f"PARCIAL_FUND2"]

    df_mat[f"INTEGRAL_INF"] = df_mat[f"INTEGRAL_CRE"] + df_mat[f"INTEGRAL_PRE"]
    df_mat[f"PARCIAL_INF"] = df_mat[f"PARCIAL_CRE"] + df_mat[f"PARCIAL_PRE"]

    for etapa in ["INF", "PRE", "CRE", "FUND", "FUND1", "FUND2", "MED"]:
        df_mat[f"TOTAL_{etapa}"] = (
            df_mat[f"INTEGRAL_{etapa}"] + df_mat[f"PARCIAL_{etapa}"]
        )

    if mod is not None:
        df_mat = df_mat[df_mat.IN_INT == mod]

    return df_mat, geo_df


@st.cache_data
def get_relacao_df_ano(df):
    return {
        "2021": df[df["ANO"] == "2021"],
        "2022": df[df["ANO"] == "2022"],
        "2023": df[df["ANO"] == "2023"],
    }


tab1, tabs2, tab3 = st.tabs(["Mapas - Taxas", "Mapas - Matrículas", "Gráficos"])

with tab1:
    dep_rel = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}

    @st.cache_data
    def _get_comparacoes(ano: str, etapa: str, dep, uf, mod):
        brazil_states = {
            "Acre": "AC",
            "Alagoas": "AL",
            "Amapá": "AP",
            "Amazonas": "AM",
            "Bahia": "BA",
            "Ceará": "CE",
            "Distrito Federal": "DF",
            "Espírito Santo": "ES",
            "Goiás": "GO",
            "Maranhão": "MA",
            "Mato Grosso": "MT",
            "Mato Grosso do Sul": "MS",
            "Minas Gerais": "MG",
            "Pará": "PA",
            "Paraíba": "PB",
            "Paraná": "PR",
            "Pernambuco": "PE",
            "Piauí": "PI",
            "Rio de Janeiro": "RJ",
            "Rio Grande do Norte": "RN",
            "Rio Grande do Sul": "RS",
            "Rondônia": "RO",
            "Roraima": "RR",
            "Santa Catarina": "SC",
            "São Paulo": "SP",
            "Sergipe": "SE",
            "Tocantins": "TO",
        }

        ano_anterior = str(int(ano) - 1)
        df_atual = df_do_ano[ano]
        df_anterior = None

        try:
            df_anterior = df_do_ano[ano_anterior]
        except KeyError:
            raise Exception("Ano inválido.")

        df_atual = df_atual[df_atual.TP_DEPENDENCIA == dep]
        df_anterior = df_anterior[df_anterior.TP_DEPENDENCIA == dep]

        if brazil_states.get(uf):
            df_atual = df_atual[df_atual.SG_UF == brazil_states[uf]]
            df_anterior = df_anterior[df_anterior.SG_UF == brazil_states[uf]]

        merged_df = pd.merge(
            df_atual,
            df_anterior,
            on=["CO_MUNICIPIO"],
            how="inner",
            suffixes=(f"_{ano}", f"_{ano_anterior}"),
        )

        merged_df[f"DIF_APROV_{etapa}"] = (
            merged_df[f"APROV_{etapa}_{ano}"]
            - merged_df[f"APROV_{etapa}_{ano_anterior}"]
        )
        merged_df[f"DIF_REPROV_{etapa}"] = (
            merged_df[f"REPROV_{etapa}_{ano}"]
            - merged_df[f"REPROV_{etapa}_{ano_anterior}"]
        )
        merged_df[f"DIF_ABAND_{etapa}"] = (
            merged_df[f"ABAND_{etapa}_{ano}"]
            - merged_df[f"ABAND_{etapa}_{ano_anterior}"]
        )

        return merged_df

    @st.cache_data
    def get_df_variacao(ano: str, etapa: str, dep: int, mod: str | None, uf):
        df = _get_comparacoes(ano=ano, etapa=etapa, dep=dep, mod=mod, uf=uf)
        return df

    def _get_titulo(ano: str, taxa: str, dep: int, uf: str = None):
        titulo = f"{ano} - {taxa} - {dep_rel[dep]}"
        if uf:
            titulo += f" - {uf}"
        return titulo

    def get_colors(coluna):
        colors = ["#bd3a3f", "#ffffff", "#449c52"]
        min = coluna.min()
        max = coluna.max()

        if np.isnan(min):
            min = -10
        if np.isnan(max):
            max = 10

        values = [min, 0, max]

        norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
        normed_vals = norm(values)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "taxa", list(zip(normed_vals, colors)), N=64
        )
        return cmap, norm

    @st.cache_data
    def plot(ano: str, etapa: str, dep: int, uf: str = None, mod: str = None):
        df = get_df_variacao(ano=ano, etapa=etapa, dep=dep, mod=mod, uf=uf)

        df_plot = geo_df[geo_df["NM_UF"] == uf] if uf else geo_df
        df_plot["CD_MUN"] = df_plot["CD_MUN"].astype(int)

        df_plot = df_plot.merge(
            df, how="left", left_on="CD_MUN", right_on="CO_MUNICIPIO"
        )

        # fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 10))
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(_get_titulo(ano, etapa, dep, uf))

        gs = gridspec.GridSpec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        coluna_aprov = df_plot[f"DIF_APROV_{etapa}"]
        coluna_reprov = df_plot[f"DIF_REPROV_{etapa}"]
        coluna_aband = df_plot[f"DIF_ABAND_{etapa}"]

        cmap_aprov, norm_aprov = get_colors(coluna_aprov)
        cmap_reprov, norm_reprov = get_colors(coluna_reprov)
        cmap_aband, norm_aband = get_colors(coluna_aband)

        ax1.set_title("Aprovação")
        ax2.set_title("Reprovação")
        ax3.set_title("Abandono")

        df_plot.plot(
            ax=ax1,
            column=f"DIF_APROV_{etapa}",
            cmap=cmap_aprov,
            norm=norm_aprov,
            legend=True,
            edgecolor="black",
            linewidth=0.05,
            missing_kwds={"color": "lightgray"},
        )

        df_plot.plot(
            ax=ax2,
            column=f"DIF_REPROV_{etapa}",
            cmap=cmap_reprov,
            norm=norm_reprov,
            legend=True,
            edgecolor="black",
            linewidth=0.05,
            missing_kwds={"color": "lightgray"},
        )

        df_plot.plot(
            ax=ax3,
            column=f"DIF_ABAND_{etapa}",
            cmap=cmap_aband,
            norm=norm_aband,
            legend=True,
            edgecolor="black",
            linewidth=0.05,
            missing_kwds={"color": "lightgray"},
        )

        st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        ano_arg = st.radio("Ano", ["2022", "2023"], key=2)

    with col2:
        etapa_arg = st.selectbox(
            "Etapa",
            [
                "FUND",
                "FUND1",
                "FUND2",
                "MED",
            ],
            key="etapa_mat_key",
        )

    with col3:
        dep_arg = st.radio(
            "Tipo de dependência",
            [2, 3],
            captions=["Estadual", "Municipal"],
            key="dep_mat_key",
        )

    uf_arg = st.text_input(
        "Estado",
    )

    mod_arg = st.radio(
        "Modalidade",
        ["TUDO", "INT", "PAR"],
        captions=[
            "Todas as escolas",
            "Apenas integral",
            "Apenas parcial",
        ],
        key="mod_mat_key",
    )

    @st.cache_data
    def get_estados():
        return [
            "Acre",
            "Alagoas",
            "Amapá",
            "Amazonas",
            "Bahia",
            "Ceará",
            "Distrito Federal",
            "Espírito Santo",
            "Goiás",
            "Maranhão",
            "Mato Grosso",
            "Mato Grosso do Sul",
            "Minas Gerais",
            "Pará",
            "Paraíba",
            "Paraná",
            "Pernambuco",
            "Piauí",
            "Rio de Janeiro",
            "Rio Grande do Norte",
            "Rio Grande do Sul",
            "Rondônia",
            "Roraima",
            "Santa Catarina",
            "São Paulo",
            "Sergipe",
            "Tocantins",
        ]

    estados = get_estados()

    st.button(label="Obter mapa", key="get_map_taxa")
    if st.session_state.pop("get_map_taxa", None):
        if uf_arg and uf_arg not in estados:
            st.error("Estado inválido. Escreva o nome do estado corretamente.")
        else:

            if mod_arg == "TUDO":
                mod = None
            else:
                mod = mod_arg == "INT"

            df, geo_df = get_data(mod)
            df_do_ano = get_relacao_df_ano(df)
            plot(ano=ano_arg, etapa=etapa_arg, dep=dep_arg, uf=uf_arg, mod=mod)

with tabs2:

    class Matriculas:
        def __init__(self, df_municipio: pd.DataFrame, geo_df: gpd.GeoDataFrame):
            self.df = df_municipio
            self.df_do_ano = {
                "2021": self.df[self.df["ANO"] == "2021"],
                "2022": self.df[self.df["ANO"] == "2022"],
                "2023": self.df[self.df["ANO"] == "2023"],
            }
            self.geo_df = geo_df
            self.dep = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}

        def _get_comparacoes(
            self, ano: str, etapa: str, modalidade: str, uf, dep: int | None = None
        ):
            brazil_states = {
                "Acre": "AC",
                "Alagoas": "AL",
                "Amapá": "AP",
                "Amazonas": "AM",
                "Bahia": "BA",
                "Ceará": "CE",
                "Distrito Federal": "DF",
                "Espírito Santo": "ES",
                "Goiás": "GO",
                "Maranhão": "MA",
                "Mato Grosso": "MT",
                "Mato Grosso do Sul": "MS",
                "Minas Gerais": "MG",
                "Pará": "PA",
                "Paraíba": "PB",
                "Paraná": "PR",
                "Pernambuco": "PE",
                "Piauí": "PI",
                "Rio de Janeiro": "RJ",
                "Rio Grande do Norte": "RN",
                "Rio Grande do Sul": "RS",
                "Rondônia": "RO",
                "Roraima": "RR",
                "Santa Catarina": "SC",
                "São Paulo": "SP",
                "Sergipe": "SE",
                "Tocantins": "TO",
            }
            ano_anterior = str(int(ano) - 1)
            df_atual = self.df_do_ano[ano]
            df_anterior = None
            try:
                df_anterior = self.df_do_ano[ano_anterior]
            except KeyError:
                raise Exception("Ano inválido.")

            if modalidade is None:
                modalidade_str = "TOTAL"
            elif modalidade == True:
                modalidade_str = "INTEGRAL"
            else:
                modalidade_str = "PARCIAL"

            df_atual = df_atual[
                (df_atual.TP_DEPENDENCIA == dep) & (df_atual.SG_UF == brazil_states[uf])
            ]
            df_anterior = df_anterior[
                (df_anterior.TP_DEPENDENCIA == dep)
                & (df_anterior.SG_UF == brazil_states[uf])
            ]

            merged_df = pd.merge(
                df_atual,
                df_anterior,
                on=["CO_MUNICIPIO"],
                how="inner",
                suffixes=(f"_{ano}", f"_{ano_anterior}"),
            )

            merged_df["diferenca"] = (
                merged_df[f"{modalidade_str}_{etapa}_{ano}"]
                - merged_df[f"{modalidade_str}_{etapa}_{ano_anterior}"]
            )

            return merged_df

        def get_df_variacao(self, ano: str, etapa: str, modalidade: str, dep: int, uf):
            df = self._get_comparacoes(
                ano=ano, etapa=etapa, modalidade=modalidade, dep=dep, uf=uf
            )
            return df
            # return df.reset_index().drop("index", axis=1)

        def _get_titulo(self, ano: str, modalidade: str, dep: int, uf: str = None):
            if modalidade is None:
                modalidade = "TOTAL"
            elif modalidade == True:
                modalidade = "INTEGRAL"
            else:
                modalidade = "PARCIAL"

            titulo = f"{ano} - {modalidade} - {self.dep[dep]}"
            if uf:
                titulo += f" - {uf}"
            return titulo

        def get_colors(self, coluna):
            colors = ["#bd3a3f", "#dbc451", "#ffffff", "#5192db", "#6d36c7"]
            min = coluna.min()
            max = coluna.max()
            print(min, max)
            if np.isnan(min) or not min:
                min = -10
            if np.isnan(max) or not max:
                max = 10
            mid_min = min / 2
            mid_max = max / 2
            values = [min, mid_min, 0, mid_max, max]

            norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
            normed_vals = norm(values)

            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "taxa", list(zip(normed_vals, colors)), N=32
            )
            return cmap, norm

        def plot(self, ano: str, etapa: str, modalidade: str, dep: int, uf: str = None):
            df = self.get_df_variacao(
                ano=ano, etapa=etapa, modalidade=modalidade, dep=dep, uf=uf
            )

            df_plot = self.geo_df[self.geo_df["NM_UF"] == uf] if uf else self.geo_df
            df_plot["CD_MUN"] = df_plot["CD_MUN"].astype(int)

            df_plot = df_plot.merge(
                df, how="left", left_on="CD_MUN", right_on="CO_MUNICIPIO"
            )

            coluna = df_plot["diferenca"]
            cmap, norm = self.get_colors(coluna)

            fig, ax = plt.subplots(figsize=(10, 8))

            ax.set_title(self._get_titulo(ano, modalidade, dep, uf))

            valores = [
                df["INTEGRAL_MED_2023"].sum(),
                df["INTEGRAL_MED_2022"].sum(),
            ]
            anos = ["2023", "2022"]

            test = pd.DataFrame({"matriculas": valores, "ano": anos})
            test

            df_plot.plot(
                ax=ax,
                column="diferenca",
                cmap=cmap,
                norm=norm,
                legend=True,
                edgecolor="black",
                linewidth=0.1,
                missing_kwds={"color": "lightgrey", "hatch": "//"},
            )

            ax.axis("off")
            st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        ano_arg = st.radio("Ano", ["2022", "2023"], key=1)

    with col2:
        etapa_arg = st.selectbox(
            "Etapa",
            [
                "INF",
                "PRE",
                "CRE",
                "FUND",
                "FUND1",
                "FUND2",
                "MED",
            ],
        )

    with col3:
        dep_arg = st.radio(
            "Tipo de dependência", [2, 3], captions=["Estadual", "Municipal"]
        )

    uf_arg = st.text_input("Estado", key="uf_mat")

    mod_arg = st.radio(
        "Modalidade",
        ["TOTAL", "INTEGRAL", "PARCIAL"],
    )

    @st.cache_data
    def get_estados():
        return [
            "Acre",
            "Alagoas",
            "Amapá",
            "Amazonas",
            "Bahia",
            "Ceará",
            "Distrito Federal",
            "Espírito Santo",
            "Goiás",
            "Maranhão",
            "Mato Grosso",
            "Mato Grosso do Sul",
            "Minas Gerais",
            "Pará",
            "Paraíba",
            "Paraná",
            "Pernambuco",
            "Piauí",
            "Rio de Janeiro",
            "Rio Grande do Norte",
            "Rio Grande do Sul",
            "Rondônia",
            "Roraima",
            "Santa Catarina",
            "São Paulo",
            "Sergipe",
            "Tocantins",
        ]

    estados = get_estados()

    st.button(label="Obter mapa", key="get_map_mat")
    if st.session_state.pop("get_map_mat", None):
        if uf_arg and uf_arg not in estados:
            st.error("Estado inválido. Escreva o nome do estado corretamente.")
        else:
            # plot(ano=ano_arg, etapa=etapa_arg, dep=dep_arg, uf=uf_arg, mod=mod_arg)
            if mod_arg == "TOTAL":
                mod = None
            else:
                mod = mod_arg == "INTEGRAL"

            df_mat, geo_df = get_data_matriculas(mod)
            mat = Matriculas(df_municipio=df_mat, geo_df=geo_df)
            res = mat.plot(
                ano=ano_arg, modalidade=mod, etapa=etapa_arg, dep=dep_arg, uf=uf_arg
            )

