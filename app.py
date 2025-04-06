import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Analyse van Arresten", layout="wide", page_icon="⚖️")

@st.cache_data
def load_data():
    pad = r"C:\\Users\\Rob\\Desktop\\Fictieve_Strafzaken_Dataset_1000.csv"
    try:
        return pd.read_csv(pad)
    except FileNotFoundError:
        st.error(f"Bestand niet gevonden op: {pad}")
        return pd.DataFrame()

df = load_data()

st.title("⚖️ Analyse van Arresten")
st.caption("🧪 Deze dataset is volledig fictief en bedoeld voor demonstratiedoeleinden.")

with st.expander("ℹ️ Wat kun je hier doen?"):
    st.markdown("""
    Deze applicatie helpt je inzicht te krijgen in patronen binnen fictieve strafzaken.

    🔍 **Wat kun je verkennen?**
    - De samenhang tussen aanleiding, gedraging, gevolg en veroordeling
    - Verschillen tussen regio’s, geslacht, en delictvormen
    - Gebruik van bewijsmiddelen per zaak (ECLI)

    🛠️ **Wat kun je doen?**
    - Zelf grafieken bouwen met de DIY-tool
    - Vergelijk groepen op basis van filters
    - Visualiseer bewijsstructuren per zaak
    """)

# === SIDEBAR FILTERS ===
toon_filters = st.checkbox("Toon filters", value=True)
if toon_filters:
    st.sidebar.header("Filters")

    if "reset" not in st.session_state:
        st.session_state.reset = False

    if st.sidebar.button("Reset filters"):
        st.session_state.reset = True

    if st.session_state.reset:
        jaren = (int(df["jaartal_delict"].min()), int(df["jaartal_delict"].max()))
        gekozen_regio = df["regionale_eenheid"].unique().tolist()
        gekozen_veroordeling = df["veroordeling"].unique().tolist()
        gekozen_tl = df["tenlastelegging"].unique().tolist()
        gekozen_mo = df["modus_operandi"].unique().tolist()
        gekozen_geslacht_dader = df["geslacht_dader"].unique().tolist()
        gekozen_geslacht_slachtoffer = df["geslacht_slachtoffer"].unique().tolist()
        st.session_state.reset = False
    else:
        jaren = st.sidebar.slider("Jaar delict", int(df["jaartal_delict"].min()),
                                  int(df["jaartal_delict"].max()), (2010, 2022))
        gekozen_regio = st.sidebar.multiselect("Regionale eenheid", df["regionale_eenheid"].unique(),
                                               default=df["regionale_eenheid"].unique())
        gekozen_veroordeling = st.sidebar.multiselect("Veroordeling", df["veroordeling"].unique(),
                                                       default=df["veroordeling"].unique())
        gekozen_tl = st.sidebar.multiselect("Tenlastelegging", df["tenlastelegging"].unique(),
                                            default=df["tenlastelegging"].unique())
        gekozen_mo = st.sidebar.multiselect("Modus operandi", df["modus_operandi"].unique(),
                                            default=df["modus_operandi"].unique())
        gekozen_geslacht_dader = st.sidebar.multiselect("Geslacht dader", df["geslacht_dader"].unique(),
                                                         default=df["geslacht_dader"].unique())
        gekozen_geslacht_slachtoffer = st.sidebar.multiselect("Geslacht slachtoffer", df["geslacht_slachtoffer"].unique(),
                                                               default=df["geslacht_slachtoffer"].unique())

# === FILTEREN VAN DATA ===
filtered = df[
    (df["jaartal_delict"] >= jaren[0]) & (df["jaartal_delict"] <= jaren[1]) &
    (df["regionale_eenheid"].isin(gekozen_regio)) &
    (df["veroordeling"].isin(gekozen_veroordeling)) &
    (df["tenlastelegging"].isin(gekozen_tl)) &
    (df["modus_operandi"].isin(gekozen_mo)) &
    (df["geslacht_dader"].isin(gekozen_geslacht_dader)) &
    (df["geslacht_slachtoffer"].isin(gekozen_geslacht_slachtoffer))
]

# === STOPKNOP ===
if st.sidebar.button("⛔ Stop de app"):
    st.warning("Applicatie wordt afgesloten...")
    os._exit(0)

# === TABS OPZETTEN ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚖️ Groepsvergelijking", 
    "📊 Algemene Analyses", 
    "🔍 Bewijslast", 
    "🛠️ DIY Visualisaties", 
    "📄 Lijst van ECLI's"])

# === TAB 1: GROEPSVERGELIJKING ===
with tab1:
    st.markdown("### ⚖️ Vergelijk twee groepen op basis van een variabele")

    vergelijk_dimensie = st.selectbox("Kies variabele om groepen te splitsen:", [
        "regionale_eenheid", "moord_of_doodslag", "tenlastelegging"])

    alle_optie = f"ALLE {vergelijk_dimensie.upper()}"
    keuzewaarden = [alle_optie] + sorted(df[vergelijk_dimensie].dropna().unique().tolist())

    groep_links = st.selectbox("Groep 1 (linkerkant)", options=keuzewaarden, key="groep_links")
    groep_rechts = st.selectbox("Groep 2 (rechterkant)", options=keuzewaarden, key="groep_rechts")

    def filter_op_groep(data, waarde, kolom):
        if waarde.startswith("ALLE"):
            return data.copy()
        return data[data[kolom] == waarde].copy()

    data_links = filter_op_groep(filtered, groep_links, vergelijk_dimensie)
    data_rechts = filter_op_groep(filtered, groep_rechts, vergelijk_dimensie)

    st.markdown(f"#### 🟦 Groep 1: {groep_links} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 🟥 Groep 2: {groep_rechts}")

    with st.expander("📅 Algemeen: jaartallen en delictinfo"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"##### 🟦 {groep_links}")
            fig1 = px.histogram(data_links, x="jaartal_delict", nbins=20)
            st.plotly_chart(fig1, use_container_width=True, key="g1_jaar_delict")
            st.plotly_chart(px.pie(data_links, names="tenlastelegging"), use_container_width=True, key="g1_tenlastelegging")
        with col2:
            st.markdown(f"##### 🟥 {groep_rechts}")
            st.plotly_chart(px.histogram(data_rechts, x="jaartal_delict", nbins=20), use_container_width=True, key="g2_jaar_delict")
            st.plotly_chart(px.pie(data_rechts, names="tenlastelegging"), use_container_width=True, key="g2_tenlastelegging")

    with st.expander("👤 Betrokkenen"):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.histogram(data_links, x="geslacht_dader", color="geslacht_slachtoffer", barmode="group"),
                              use_container_width=True, key="g1_geslacht")
        with col2:
            st.plotly_chart(px.histogram(data_rechts, x="geslacht_dader", color="geslacht_slachtoffer", barmode="group"),
                              use_container_width=True, key="g2_geslacht")

    with st.expander("⚖️ Juridisch"):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.box(data_links, y="strafmaat"), use_container_width=True, key="g1_strafmaat")
        with col2:
            st.plotly_chart(px.box(data_rechts, y="strafmaat"), use_container_width=True, key="g2_strafmaat")

# === TAB 2: ALGEMENE ANALYSES ===
with tab2:
    st.markdown("### 📊 Algemene Analyses")

    with st.expander("📅 Algemeen"):
        st.plotly_chart(px.histogram(filtered, x="jaartal_delict", nbins=20),
                         use_container_width=True, key="alg_jaartal_delict")
        st.plotly_chart(px.histogram(filtered, x="jaartal_uitspraak", nbins=20),
                         use_container_width=True, key="alg_jaartal_uitspraak")
        st.plotly_chart(px.pie(filtered, names="tenlastelegging"),
                         use_container_width=True, key="alg_pie_tl")
        st.plotly_chart(px.bar(filtered, x="modus_operandi"),
                         use_container_width=True, key="alg_mo_bar")

    with st.expander("👤 Betrokkenen"):
        st.plotly_chart(px.histogram(filtered, x="geslacht_dader", color="geslacht_slachtoffer", barmode="group"),
                         use_container_width=True, key="alg_geslacht")

    with st.expander("⚖️ Juridisch"):
        st.plotly_chart(px.histogram(filtered, x="veroordeling"),
                         use_container_width=True, key="alg_veroordeling_hist")
        st.plotly_chart(px.box(filtered, y="strafmaat"),
                         use_container_width=True, key="alg_strafmaat_box")
        st.plotly_chart(px.pie(filtered, names="verweer_verdediging"),
                         use_container_width=True, key="alg_pie_verweer")
        st.plotly_chart(px.bar(filtered, x="bewijstypen"),
                         use_container_width=True, key="alg_bewijs_bar")

    with st.expander("🌍 Regio"):
        st.plotly_chart(px.histogram(filtered, x="regionale_eenheid", color="moord_of_doodslag", barmode="group"),
                         use_container_width=True, key="alg_regio_hist")

    with st.expander("✨ Extra visualisaties"):
        st.plotly_chart(px.treemap(filtered, path=['veroordeling', 'bewijstypen'], title="Treemap"),
                         use_container_width=True, key="alg_treemap")
        st.plotly_chart(px.violin(filtered, x='geslacht_dader', y='strafmaat', box=True, points='all'),
                         use_container_width=True, key="alg_violin")
        st.plotly_chart(px.density_heatmap(filtered, x='jaartal_delict', y='jaartal_uitspraak'),
                         use_container_width=True, key="alg_heatmap")

        st.markdown("#### 🔁 Sankey-diagram: Van Aanleiding tot Juridische Kwalificatie")

        if all(col in filtered.columns for col in ["aanleiding", "gedraging", "gevolg", "veroordeling"]):
            sankey_df = filtered.groupby(["aanleiding", "gedraging", "gevolg", "veroordeling"]).size().reset_index(name="aantal")
            labels = list(pd.unique(sankey_df[["aanleiding", "gedraging", "gevolg", "veroordeling"]].values.ravel()))
            label_to_index = {label: i for i, label in enumerate(labels)}

            def create_links(source_col, target_col):
                return [
                    (label_to_index[row[source_col]], label_to_index[row[target_col]], row["aantal"])
                    for _, row in sankey_df.iterrows()
                ]

            links = create_links("aanleiding", "gedraging") + create_links("gedraging", "gevolg") + create_links("gevolg", "veroordeling")
            source, target, value = zip(*links)

            fig = go.Figure(data=[go.Sankey(
                node=dict(label=labels, pad=15, thickness=20),
                link=dict(source=source, target=target, value=value)
            )])

            fig.update_layout(title_text="Stroom van Aanleiding → Gedraging → Gevolg → Juridische kwalificatie", font_size=12)
            st.plotly_chart(fig, use_container_width=True, key="alg_sankey")
        else:
            st.warning("De vereiste kolommen 'aanleiding', 'gedraging', 'gevolg' en 'veroordeling' ontbreken in de dataset.")

# === TAB 3: BEWIJSLAST ===
with tab3:
    st.markdown("### 🔍 Bewijslast per ECLI")

    bewijs_path = r"C:\\Users\\Rob\\Desktop\\bewijsmiddelen_ai_all_hersteld.xlsx"
    try:
        bewijs_df = pd.read_excel(bewijs_path)
        bewijs_df.columns = bewijs_df.columns.str.strip()
        for kolom in ['ECLI', 'Hoofdcategorie', 'Subcategorie']:
            bewijs_df[kolom] = bewijs_df[kolom].astype(str).str.strip()
            bewijs_df[kolom] = bewijs_df[kolom].replace(['', 'nan', 'NaN'], pd.NA)
        bewijs_df[['ECLI', 'Hoofdcategorie', 'Subcategorie']] = bewijs_df[['ECLI', 'Hoofdcategorie', 'Subcategorie']].ffill()
        bewijs_df = bewijs_df.dropna(subset=['ECLI', 'Hoofdcategorie', 'Subcategorie'])
        bewijs_df = bewijs_df[(bewijs_df['Hoofdcategorie'] != '') & (bewijs_df['Subcategorie'] != '')]

        unieke_ecli = sorted(bewijs_df['ECLI'].unique())
        ecli_selectie = st.selectbox("Selecteer ECLI", options=["Totaaloverzicht", "Totaaloverzicht 2 (unieke zaken)"] + unieke_ecli)

        unieke_sub = sorted(bewijs_df['Subcategorie'].unique())
        unieke_hoofd = sorted(bewijs_df['Hoofdcategorie'].unique())

        if ecli_selectie == "Totaaloverzicht":
            heatmap_data = pd.DataFrame(0, index=unieke_sub, columns=unieke_hoofd)
            for _, row in bewijs_df.iterrows():
                sub = row['Subcategorie']
                hoofd = row['Hoofdcategorie']
                if sub in heatmap_data.index and hoofd in heatmap_data.columns:
                    heatmap_data.loc[sub, hoofd] += 1

        elif ecli_selectie == "Totaaloverzicht 2 (unieke zaken)":
            unieke_zaken = bewijs_df[['ECLI', 'Subcategorie', 'Hoofdcategorie']].drop_duplicates()
            unieke_zaken['waarde'] = 1
            heatmap_data = pd.pivot_table(
                unieke_zaken,
                index='Subcategorie',
                columns='Hoofdcategorie',
                values='waarde',
                aggfunc='sum',
                fill_value=0
            )
        else:
            gefilterd = bewijs_df[bewijs_df['ECLI'] == ecli_selectie]
            heatmap_data = pd.DataFrame(0, index=unieke_sub, columns=unieke_hoofd)
            for _, row in gefilterd.iterrows():
                sub = row['Subcategorie']
                hoofd = row['Hoofdcategorie']
                if sub in heatmap_data.index and hoofd in heatmap_data.columns:
                    heatmap_data.loc[sub, hoofd] += 1

        customdata = [[f"{y}|||{x}" for x in heatmap_data.columns] for y in heatmap_data.index]

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='YlGnBu',
            colorbar=dict(title="Aantal"),
            customdata=customdata,
            hovertemplate="<b>Subcategorie</b>: %{y}<br><b>Hoofdcategorie</b>: %{x}<br><b>Aantal</b>: %{z}<extra></extra>"
        ))

        fig.update_layout(
            font=dict(family="Calibri", size=12),
            xaxis_title="Hoofdcategorie",
            yaxis_title="Subcategorie",
            xaxis=dict(tickangle=45),
            margin=dict(l=80, r=20, t=60, b=60),
            height=700
        )

        st.plotly_chart(fig, use_container_width=True, key="bewijs_heatmap")

    except FileNotFoundError:
        st.warning("Excelbestand met bewijsmiddelen niet gevonden.")

# === TAB 4: DIY VISUALISATIES ===
with tab4:
    st.markdown("### 🛠️ Speel zelf met de visualisaties")
    with st.expander("ℹ️ Uitleg DIY-gedeelte"):
        st.markdown("""
        Hier kun je zelf visualisaties samenstellen:
        - Kies het gewenste type grafiek
        - Selecteer de variabelen voor de assen, kleur en facetten
        - Experimenteer met combinaties om verbanden te ontdekken

        Bijvoorbeeld:
        - Bekijk of 'modus_operandi' verschilt per 'geslacht_dader'
        - Splits op 'regionale_eenheid' om verschillen per regio te zien
        """)
    beschikbare_kolommen = filtered.columns.tolist()
    facet_dimensie = st.selectbox("Facet naar variabele (optioneel)", [None] + beschikbare_kolommen)
    if facet_dimensie:
        unieke_facetwaarden = sorted(df[facet_dimensie].dropna().unique().tolist())
        gekozen_facetwaarden = st.multiselect(f"Selecteer waarden van '{facet_dimensie}' om mee te nemen", unieke_facetwaarden, default=unieke_facetwaarden)
        filtered = filtered[filtered[facet_dimensie].isin(gekozen_facetwaarden)]
        st.markdown("---")
    plot_type = st.selectbox("📊 Kies een plotsoort", ["Staafdiagram", "Histogram", "Spreidingsplot (Ballon)", "Ballonplot met facet", "Ballonplot (ggballoonplot-stijl)", "Sunburst", "Sankey"])
    if plot_type == "Staafdiagram":
        x = st.selectbox("X-as", beschikbare_kolommen, help="De variabele op de horizontale as van de grafiek")
        kleur = st.selectbox("Kleur (optioneel)", [None] + beschikbare_kolommen, help="Kleur de categorieën voor extra inzicht")
        st.plotly_chart(px.bar(filtered, x=x, color=kleur if kleur else None, facet_col=facet_dimensie, facet_col_wrap=2), use_container_width=True, key="diy_bar")
    elif plot_type == "Histogram":
        x = st.selectbox("X-as (numeriek)", [col for col in beschikbare_kolommen if pd.api.types.is_numeric_dtype(filtered[col])])
        kleur = st.selectbox("Kleur (optioneel)", [None] + beschikbare_kolommen)
        st.plotly_chart(px.histogram(filtered, x=x, color=kleur if kleur else None, facet_col=facet_dimensie, facet_col_wrap=2), use_container_width=True, key="diy_hist")
    elif plot_type == "Spreidingsplot (Ballon)":
        x = st.selectbox("X-as", beschikbare_kolommen)
        y = st.selectbox("Y-as", beschikbare_kolommen)
        grootte = st.selectbox("Grootte van de ballon", [None] + [col for col in beschikbare_kolommen if pd.api.types.is_numeric_dtype(filtered[col])])
        kleur = st.selectbox("Kleur (optioneel)", [None] + beschikbare_kolommen)
        st.plotly_chart(px.scatter(filtered, x=x, y=y, size=grootte if grootte else None, color=kleur if kleur else None, facet_col=facet_dimensie, facet_col_wrap=2), use_container_width=True, key="diy_scatter")
    elif plot_type == "Ballonplot met facet":
        x = st.selectbox("X-as", beschikbare_kolommen)
        y = st.selectbox("Y-as", beschikbare_kolommen)
        grootte = st.selectbox("Grootte van de ballon", [None] + [col for col in beschikbare_kolommen if pd.api.types.is_numeric_dtype(filtered[col])])
        kleur = st.selectbox("Kleur (optioneel)", [None] + beschikbare_kolommen)
        facet = st.selectbox("Facet (opsplitsen in kleine grafieken)", beschikbare_kolommen)
        st.plotly_chart(px.scatter(filtered, x=x, y=y, size=grootte if grootte else None, color=kleur if kleur else None, facet_col=facet, facet_col_wrap=2), use_container_width=True, key="diy_scatter_facet")
    elif plot_type == "Ballonplot (ggballoonplot-stijl)":
        x = st.selectbox("X-as (categorie)", beschikbare_kolommen)
        y = st.selectbox("Y-as (categorie)", beschikbare_kolommen)
        kleur = st.selectbox("Kleur (optioneel)", [None] + beschikbare_kolommen)
        balloon_df = filtered.groupby([x, y, facet_dimensie] if facet_dimensie else [x, y]).size().reset_index(name='Aantal')
        fig = px.scatter(balloon_df, x=x, y=y, size='Aantal', color=kleur if kleur and kleur in balloon_df.columns else None, facet_col=facet_dimensie, facet_col_wrap=2)
        st.plotly_chart(fig, use_container_width=True, key="diy_ggballoon")
    elif plot_type == "Sunburst":
        ring1 = st.selectbox("🔘 Binnenste ring", beschikbare_kolommen, key="diy_ring1")
        ring2 = st.selectbox("⭕ Middelste ring", beschikbare_kolommen, key="diy_ring2")
        ring3 = st.selectbox("🌀 Buitenste ring", beschikbare_kolommen, key="diy_ring3")
        if len({ring1, ring2, ring3}) == 3:
            st.plotly_chart(px.sunburst(filtered, path=[ring1, ring2, ring3], facet_col=facet_dimensie if facet_dimensie else None), use_container_width=True, key="diy_sunburst")
        else:
            st.warning("Kies drie verschillende variabelen.")
    elif plot_type == "Sankey":
        bron = st.selectbox("Bron (source)", beschikbare_kolommen, key="diy_sankey_src")
        doel = st.selectbox("Doel (target)", beschikbare_kolommen, key="diy_sankey_tgt")
        if bron != doel:
            if facet_dimensie:
                for waarde in sorted(filtered[facet_dimensie].dropna().unique()):
                    deel_df = filtered[filtered[facet_dimensie] == waarde]
                    df_sankey = deel_df.groupby([bron, doel]).size().reset_index(name="aantal")
                    labels = list(pd.unique(df_sankey[[bron, doel]].values.ravel()))
                    label_map = {label: i for i, label in enumerate(labels)}
                    source = df_sankey[bron].map(label_map)
                    target = df_sankey[doel].map(label_map)
                    value = df_sankey["aantal"]
                    fig = go.Figure(data=[go.Sankey(
                        node=dict(label=labels, pad=15, thickness=20),
                        link=dict(source=source, target=target, value=value)
                    )])
                    fig.update_layout(title_text=f"Sankey: {bron} → {doel} | {facet_dimensie} = {waarde}", font_size=12)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                df_sankey = filtered.groupby([bron, doel]).size().reset_index(name="aantal")
            labels = list(pd.unique(df_sankey[[bron, doel]].values.ravel()))
            label_map = {label: i for i, label in enumerate(labels)}
            source = df_sankey[bron].map(label_map)
            target = df_sankey[doel].map(label_map)
            value = df_sankey["aantal"]
            fig = go.Figure(data=[go.Sankey(
                node=dict(label=labels, pad=15, thickness=20),
                link=dict(source=source, target=target, value=value)
            )])
            fig.update_layout(title_text=f"Sankey: {bron} → {doel}", font_size=12)
            st.plotly_chart(fig, use_container_width=True, key="diy_sankey_final")
        else:
            st.warning("Kies twee verschillende kolommen.")

# === TAB 5: LIJST VAN ECLI'S ===
with tab5:
    st.markdown("### 📄 Lijst van ECLI's met filtermogelijkheden")
    if 'ECLI' in df.columns:
        ecli_df = df.copy()
        with st.expander("🔎 Filters toepassen op ECLI-lijst"):
            for kolom in df.columns:
                unieke_waarden = sorted(df[kolom].dropna().unique().tolist())
                if len(unieke_waarden) <= 30 and df[kolom].dtype == object:
                    geselecteerd = st.multiselect(f"{kolom}", unieke_waarden, default=unieke_waarden)
                    ecli_df = ecli_df[ecli_df[kolom].isin(geselecteerd)]
                elif pd.api.types.is_numeric_dtype(df[kolom]):
                    minval, maxval = int(df[kolom].min()), int(df[kolom].max())
                    gekozen_range = st.slider(f"{kolom}", minval, maxval, (minval, maxval))
                    ecli_df = ecli_df[(df[kolom] >= gekozen_range[0]) & (df[kolom] <= gekozen_range[1])]
        st.markdown(f"**Aantal resultaten: {len(ecli_df)}**")
        st.dataframe(ecli_df[['ECLI']].drop_duplicates().sort_values(by='ECLI').reset_index(drop=True), use_container_width=True)
    else:
        st.warning("Er is geen kolom 'ECLI' in de dataset gevonden.")
