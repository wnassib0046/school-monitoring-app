import os
import sqlite3
import unicodedata
from datetime import datetime, date

import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# =========================
# CONFIGURATION
# =========================

st.set_page_config(
    page_title="Application de Suivi Scolaire",
    page_icon="📊",
    layout="wide"
)

FILE_PATH = r"data\analysis_all_classes1.xlsx"
SUIVI_PATH = r"data\suivi_eleves.xlsx"
DB_PATH = r"database\suivi.db"

# =========================
# STYLE
# =========================

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 16px;
    color: #9ca3af;
    margin-bottom: 28px;
}

.section-title {
    font-size: 24px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 16px;
}

.kpi-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #1f2937;
    text-align: center;
    min-height: 115px;
}

.kpi-label {
    font-size: 14px;
    color: #9ca3af;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 30px;
    font-weight: 850;
    color: #ffffff;
}

.info-card {
    background-color: #111827;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #1f2937;
    margin-bottom: 10px;
}

.info-label {
    font-size: 13px;
    color: #9ca3af;
}

.info-value {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
}

.block-container {
    padding-top: 2rem;
}

hr {
    margin-top: 25px;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# HELPERS
# =========================

def clean_column_name(col):
    """Convertit les noms de colonnes en format simple sans accents."""
    col = str(col).strip().lower()
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace("/", "_")
    while "__" in col:
        col = col.replace("__", "_")
    return col


def normalize_columns(df):
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def safe_column(df, column_name, default_value=""):
    if column_name not in df.columns:
        df[column_name] = default_value
    return df


def make_kpi(label, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def make_info(label, value):
    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">{label}</div>
        <div class="info-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def apply_student_filters(df, key_prefix="main"):
    filtered = df.copy()

    classes = ["Toutes"] + sorted(filtered["classe"].dropna().unique().tolist())
    classe_selectionnee = st.sidebar.selectbox(
        "Classe",
        classes,
        key=f"{key_prefix}_classe"
    )

    statuts_scolaires = ["Tous"] + sorted(filtered["statut_scolaire"].dropna().unique().tolist())
    statut_scolaire = st.sidebar.selectbox(
        "Statut scolaire",
        statuts_scolaires,
        key=f"{key_prefix}_statut_scolaire"
    )

    statuts_finaux = ["Tous"] + sorted(filtered["statut_final"].dropna().unique().tolist())
    statut_final = st.sidebar.selectbox(
        "Statut final",
        statuts_finaux,
        key=f"{key_prefix}_statut_final"
    )

    if classe_selectionnee != "Toutes":
        filtered = filtered[filtered["classe"] == classe_selectionnee]

    if statut_scolaire != "Tous":
        filtered = filtered[filtered["statut_scolaire"] == statut_scolaire]

    if statut_final != "Tous":
        filtered = filtered[filtered["statut_final"] == statut_final]

    return filtered, classe_selectionnee


def get_status_counts(df):
    if df.empty:
        return pd.DataFrame(columns=["statut_scolaire", "nombre_eleves"])

    result = df["statut_scolaire"].value_counts().reset_index()
    result.columns = ["statut_scolaire", "nombre_eleves"]
    return result


def load_suivi():
    if not os.path.exists(SUIVI_PATH):
        suivi = pd.DataFrame(columns=[
            "date_suivi",
            "id_eleve",
            "nom_complet",
            "classe",
            "type_action",
            "observation",
            "responsable",
            "prochaine_action",
            "statut_suivi"
        ])
        suivi.to_excel(SUIVI_PATH, index=False)

    return pd.read_excel(SUIVI_PATH)


def save_suivi(df):
    df.to_excel(SUIVI_PATH, index=False)

def init_db():
    os.makedirs("database", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS suivi_eleves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_suivi TEXT,
            id_eleve TEXT,
            nom_complet TEXT,
            classe TEXT,
            type_action TEXT,
            observation TEXT,
            responsable TEXT,
            prochaine_action TEXT,
            statut_suivi TEXT,
            date_creation TEXT
        )
    """)

    conn.commit()
    conn.close()


def ajouter_suivi(date_suivi, id_eleve, nom_complet, classe, type_action,
                  observation, responsable, prochaine_action, statut_suivi):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO suivi_eleves (
            date_suivi,
            id_eleve,
            nom_complet,
            classe,
            type_action,
            observation,
            responsable,
            prochaine_action,
            statut_suivi,
            date_creation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(date_suivi),
        str(id_eleve),
        str(nom_complet),
        str(classe),
        str(type_action),
        str(observation),
        str(responsable),
        str(prochaine_action),
        str(statut_suivi),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def charger_suivi():
    conn = sqlite3.connect(DB_PATH)

    try:
        suivi = pd.read_sql_query(
            "SELECT * FROM suivi_eleves ORDER BY date_creation DESC",
            conn
        )
    except Exception:
        suivi = pd.DataFrame()

    conn.close()
    return suivi

def compter_matieres_faibles(valeur):
    if pd.isna(valeur):
        return 0

    valeur = str(valeur).strip()

    if valeur.lower() in ["aucune", "nan", "", "none"]:
        return 0

    return len([m for m in valeur.split(",") if m.strip() != ""])


def creer_modele_risque(df):
    data = df.copy()

    data["nombre_matieres_faibles"] = data["matieres_faibles"].apply(compter_matieres_faibles)

    data = data.dropna(subset=[
        "moyenne_generale",
        "total_absences",
        "nombre_matieres_faibles",
        "statut_final"
    ])

    features = [
        "moyenne_generale",
        "total_absences",
        "nombre_matieres_faibles"
    ]

    X = data[features]

    y = data["statut_final"].apply(
        lambda x: "Risque élevé" if x == "À risque"
        else "Risque moyen" if x in ["Faible", "À suivre"]
        else "Risque faible"
    )

    if len(data) < 10 or y.nunique() < 2:
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    predictions = model.predict(X)

    data["risque_predit"] = predictions

    return model, data, accuracy, X_test, y_test
# =========================
# DATA LOADING
# =========================

@st.cache_data
def load_data():
    etudiants = pd.read_excel(FILE_PATH, sheet_name="Etudiants")
    synthese_eleves = pd.read_excel(FILE_PATH, sheet_name="Synthese_eleves")
    synthese_classes = pd.read_excel(FILE_PATH, sheet_name="Synthese_classes")
    absences_mensuelles = pd.read_excel(FILE_PATH, sheet_name="Absences_mensuelles")
    synthese_matieres = pd.read_excel(FILE_PATH, sheet_name="Synthese_matieres")
    notes = pd.read_excel(FILE_PATH, sheet_name="Notes")
    absences = pd.read_excel(FILE_PATH, sheet_name="Absences")

    etudiants = normalize_columns(etudiants)
    synthese_eleves = normalize_columns(synthese_eleves)
    synthese_classes = normalize_columns(synthese_classes)
    absences_mensuelles = normalize_columns(absences_mensuelles)
    synthese_matieres = normalize_columns(synthese_matieres)
    notes = normalize_columns(notes)
    absences = normalize_columns(absences)

    return (
        etudiants,
        synthese_eleves,
        synthese_classes,
        absences_mensuelles,
        synthese_matieres,
        notes,
        absences
    )


# =========================
# APP HEADER
# =========================

st.markdown('<div class="main-title">📊 Application de Suivi Scolaire</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Tableau de bord pour le suivi des notes, des absences et des élèves à risque</div>',
    unsafe_allow_html=True
)


# =========================
# MAIN APP
# =========================

try:
    (
        etudiants,
        synthese_eleves,
        synthese_classes,
        absences_mensuelles,
        synthese_matieres,
        notes,
        absences
    ) = load_data()
    (
    etudiants,
    synthese_eleves,
    synthese_classes,
    absences_mensuelles,
    synthese_matieres,
    notes,
    absences
    ) = load_data()

    init_db()
    # Colonnes importantes
    synthese_eleves = safe_column(synthese_eleves, "statut_scolaire", "Non défini")
    synthese_eleves = safe_column(synthese_eleves, "statut_final", "Non défini")
    synthese_eleves = safe_column(synthese_eleves, "matieres_faibles", "Aucune")
    synthese_eleves = safe_column(synthese_eleves, "recommandation", "Aucune recommandation")

    st.sidebar.title("Navigation")

    if st.sidebar.button("🔄 Actualiser les données"):
        st.cache_data.clear()
        st.rerun()

    page = st.sidebar.radio(
        "Choisir une page",
        [
            "Tableau de bord",
            "Élèves",
            "Profil élève",
            "Analyse des notes",
            "Analyse des absences",
            "Élèves faibles",
            "Prédiction du risque",
            "Suivi élèves",
            "Qualité des données",
            "À propos du projet"
        ]
    )

    # =========================
    # PAGE 1: DASHBOARD
    # =========================

    if page == "Tableau de bord":
        st.success("✅ Données chargées avec succès")

        st.sidebar.markdown("---")
        st.sidebar.header("Filtres")
        df_filtre, classe_selectionnee = apply_student_filters(synthese_eleves, "dashboard")

        if df_filtre.empty:
            st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        else:
            total_eleves = len(df_filtre)
            moyenne_generale = df_filtre["moyenne_generale"].mean()
            total_absences = df_filtre["total_absences"].sum()
            eleves_faibles = len(df_filtre[df_filtre["statut_scolaire"].isin(["À risque", "Faible"])])
            eleves_risque = len(df_filtre[df_filtre["statut_final"] == "À risque"])

            st.markdown('<div class="section-title">📌 Indicateurs principaux</div>', unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                make_kpi("Total élèves", total_eleves)

            with col2:
                make_kpi("Moyenne générale", f"{moyenne_generale:.2f}/10")

            with col3:
                make_kpi("Total absences", int(total_absences))

            with col4:
                make_kpi("Élèves faibles", eleves_faibles)

            with col5:
                make_kpi("Élèves à risque", eleves_risque)

            st.divider()

            tab1, tab2, tab3 = st.tabs([
                "Vue générale",
                "Absences",
                "Statuts"
            ])

            with tab1:
                st.markdown('<div class="section-title">📊 Analyse par classe</div>', unsafe_allow_html=True)

                resume_classes_filtre = df_filtre.groupby("classe").agg(
                    moyenne_classe=("moyenne_generale", "mean"),
                    total_absences=("total_absences", "sum"),
                    nombre_eleves=("id_eleve", "count")
                ).reset_index()

                col_graph1, col_graph2 = st.columns(2)

                with col_graph1:
                    fig_moyenne = px.bar(
                        resume_classes_filtre,
                        x="classe",
                        y="moyenne_classe",
                        text="moyenne_classe",
                        title="Moyenne générale par classe"
                    )
                    fig_moyenne.update_traces(
                        texttemplate="%{text:.2f}",
                        textposition="outside"
                    )
                    fig_moyenne.update_layout(
                        xaxis_title="Classe",
                        yaxis_title="Moyenne /10",
                        height=450
                    )
                    st.plotly_chart(fig_moyenne, width="stretch")

                with col_graph2:
                    fig_absences = px.bar(
                        resume_classes_filtre,
                        x="classe",
                        y="total_absences",
                        text="total_absences",
                        title="Total des absences par classe"
                    )
                    fig_absences.update_traces(
                        texttemplate="%{text:.0f}",
                        textposition="outside"
                    )
                    fig_absences.update_layout(
                        xaxis_title="Classe",
                        yaxis_title="Total des absences",
                        height=450
                    )
                    st.plotly_chart(fig_absences, width="stretch")

                with st.expander("Voir le tableau détaillé des élèves"):
                    st.write(f"Nombre d'élèves affichés : {len(df_filtre)}")
                    st.dataframe(df_filtre, width="stretch")

            with tab2:
                st.markdown('<div class="section-title">📅 Analyse des absences</div>', unsafe_allow_html=True)

                absences_filtrees = absences_mensuelles.copy()

                if classe_selectionnee != "Toutes":
                    absences_filtrees = absences_filtrees[absences_filtrees["classe"] == classe_selectionnee]

                if "numero_mois" in absences_filtrees.columns:
                    mois_col = "numero_mois"
                elif "numro_mois" in absences_filtrees.columns:
                    mois_col = "numro_mois"
                else:
                    mois_col = "mois"

                resume_absences_mois = absences_filtrees.groupby(["mois", mois_col]).agg(
                    total_absences=("total_absences", "sum")
                ).reset_index()

                resume_absences_mois = resume_absences_mois.sort_values(mois_col)

                fig_absences_mois = px.line(
                    resume_absences_mois,
                    x="mois",
                    y="total_absences",
                    markers=True,
                    text="total_absences",
                    title="Évolution mensuelle des absences"
                )
                fig_absences_mois.update_traces(textposition="top center")
                fig_absences_mois.update_layout(
                    xaxis_title="Mois",
                    yaxis_title="Total des absences",
                    xaxis=dict(type="category"),
                    height=450
                )
                st.plotly_chart(fig_absences_mois, width="stretch")

                st.markdown('<div class="section-title">🚨 Top 10 élèves les plus absents</div>', unsafe_allow_html=True)

                top_absents = df_filtre.sort_values(
                    by="total_absences",
                    ascending=False
                ).head(10)

                fig_top_absents = px.bar(
                    top_absents,
                    x="total_absences",
                    y="nom_complet",
                    orientation="h",
                    text="total_absences",
                    title="Top 10 élèves les plus absents"
                )
                fig_top_absents.update_traces(textposition="outside")
                fig_top_absents.update_layout(
                    xaxis_title="Total des absences",
                    yaxis_title="Élève",
                    yaxis=dict(autorange="reversed"),
                    height=500
                )
                st.plotly_chart(fig_top_absents, width="stretch")

                st.dataframe(
                    top_absents[[
                        "id_eleve",
                        "nom_complet",
                        "classe",
                        "moyenne_generale",
                        "total_absences",
                        "statut_final",
                        "recommandation"
                    ]],
                    width="stretch"
                )

            with tab3:
                st.markdown('<div class="section-title">📌 Répartition des statuts</div>', unsafe_allow_html=True)

                statut_count = get_status_counts(df_filtre)

                fig_statut = px.pie(
                    statut_count,
                    names="statut_scolaire",
                    values="nombre_eleves",
                    title="Répartition des élèves par statut scolaire"
                )
                st.plotly_chart(fig_statut, width="stretch")

    # =========================
    # PAGE 2: STUDENTS
    # =========================

    elif page == "Élèves":
        st.header("👨‍🎓 Liste des élèves")

        col_f1, col_f2, col_f3 = st.columns(3)

        df_etudiants = synthese_eleves.copy()

        with col_f1:
            recherche = st.text_input("Rechercher un élève")

        with col_f2:
            classes_etudiants = ["Toutes"] + sorted(df_etudiants["classe"].dropna().unique().tolist())
            classe_etudiant = st.selectbox("Filtrer par classe", classes_etudiants)

        with col_f3:
            statuts_etudiants = ["Tous"] + sorted(df_etudiants["statut_final"].dropna().unique().tolist())
            statut_etudiant = st.selectbox("Filtrer par statut final", statuts_etudiants)

        if classe_etudiant != "Toutes":
            df_etudiants = df_etudiants[df_etudiants["classe"] == classe_etudiant]

        if statut_etudiant != "Tous":
            df_etudiants = df_etudiants[df_etudiants["statut_final"] == statut_etudiant]

        if recherche:
            df_etudiants = df_etudiants[
                df_etudiants["nom_complet"].str.contains(recherche, case=False, na=False)
            ]

        st.write(f"Nombre d'élèves trouvés : {len(df_etudiants)}")

        st.dataframe(
            df_etudiants[[
                "id_eleve",
                "nom_complet",
                "classe",
                "niveau",
                "moyenne_generale",
                "total_absences",
                "statut_scolaire",
                "statut_final",
                "matieres_faibles",
                "recommandation"
            ]],
            width="stretch"
        )

    # =========================
    # PAGE 3: STUDENT PROFILE
    # =========================

    elif page == "Profil élève":
        st.header("👤 Profil d'un élève")

        classes_profil = ["Toutes"] + sorted(synthese_eleves["classe"].dropna().unique().tolist())
        classe_profil = st.selectbox("Filtrer par classe", classes_profil)

        df_profil = synthese_eleves.copy()

        if classe_profil != "Toutes":
            df_profil = df_profil[df_profil["classe"] == classe_profil]

        liste_eleves = sorted(df_profil["nom_complet"].dropna().unique().tolist())

        if not liste_eleves:
            st.warning("Aucun élève disponible.")
        else:
            eleve_selectionne = st.selectbox("Sélectionner un élève", liste_eleves)

            profil_eleve = df_profil[df_profil["nom_complet"] == eleve_selectionne].iloc[0]

            col_p1, col_p2, col_p3, col_p4 = st.columns(4)

            with col_p1:
                make_kpi("Classe", profil_eleve["classe"])

            with col_p2:
                make_kpi("Moyenne", f"{profil_eleve['moyenne_generale']:.2f}/10")

            with col_p3:
                make_kpi("Absences", int(profil_eleve["total_absences"]))

            with col_p4:
                make_kpi("Statut final", profil_eleve["statut_final"])

            st.divider()

            col_info1, col_info2 = st.columns(2)

            with col_info1:
                make_info("Nom complet", profil_eleve["nom_complet"])
                make_info("Niveau", profil_eleve["niveau"])
                make_info("Statut scolaire", profil_eleve["statut_scolaire"])

            with col_info2:
                make_info("Matières faibles", profil_eleve["matieres_faibles"])
                make_info("Recommandation", profil_eleve["recommandation"])

            st.divider()

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.subheader("📊 Notes par matière")

                notes_eleve = notes[notes["id_eleve"] == profil_eleve["id_eleve"]]

                if notes_eleve.empty:
                    st.info("Aucune note disponible pour cet élève.")
                else:
                    notes_eleve_resume = notes_eleve.groupby("matiere_principale").agg(
                        moyenne=("note", "mean")
                    ).reset_index()

                    fig_notes_eleve = px.bar(
                        notes_eleve_resume,
                        x="matiere_principale",
                        y="moyenne",
                        text="moyenne",
                        title="Moyenne par matière"
                    )
                    fig_notes_eleve.update_traces(
                        texttemplate="%{text:.2f}",
                        textposition="outside"
                    )
                    fig_notes_eleve.update_layout(
                        xaxis_title="Matière",
                        yaxis_title="Moyenne /10",
                        height=450
                    )
                    st.plotly_chart(fig_notes_eleve, width="stretch")

            with col_chart2:
                st.subheader("📅 Absences mensuelles")

                absences_eleve = absences[absences["id_eleve"] == profil_eleve["id_eleve"]]

                if absences_eleve.empty:
                    st.info("Aucune absence disponible pour cet élève.")
                else:
                    if "numero_mois" in absences_eleve.columns:
                        absences_eleve = absences_eleve.sort_values("numero_mois")
                    elif "numro_mois" in absences_eleve.columns:
                        absences_eleve = absences_eleve.sort_values("numro_mois")

                    fig_absences_eleve = px.line(
                        absences_eleve,
                        x="mois",
                        y="total_absences",
                        markers=True,
                        text="total_absences",
                        title="Évolution des absences"
                    )
                    fig_absences_eleve.update_traces(textposition="top center")
                    fig_absences_eleve.update_layout(
                        xaxis_title="Mois",
                        yaxis_title="Absences",
                        xaxis=dict(type="category"),
                        height=450
                    )
                    st.plotly_chart(fig_absences_eleve, width="stretch")

    # =========================
    # PAGE 4: GRADES ANALYSIS
    # =========================

    elif page == "Analyse des notes":
        st.header("📚 Analyse des notes et matières")

        classes_notes = ["Toutes"] + sorted(synthese_matieres["classe"].dropna().unique().tolist())
        classe_notes = st.selectbox("Filtrer par classe", classes_notes)

        matieres_filtrees = synthese_matieres.copy()

        if classe_notes != "Toutes":
            matieres_filtrees = matieres_filtrees[matieres_filtrees["classe"] == classe_notes]

        if matieres_filtrees.empty:
            st.warning("Aucune donnée disponible.")
        else:
            col_n1, col_n2 = st.columns(2)

            with col_n1:
                st.subheader("📉 Matières les plus faibles")

                top_matieres_faibles = matieres_filtrees.sort_values(
                    by="moyenne",
                    ascending=True
                ).head(10)

                fig_matieres_faibles = px.bar(
                    top_matieres_faibles,
                    x="moyenne",
                    y="matiere_principale",
                    orientation="h",
                    text="moyenne",
                    title="Top 10 matières faibles"
                )
                fig_matieres_faibles.update_traces(
                    texttemplate="%{text:.2f}",
                    textposition="outside"
                )
                fig_matieres_faibles.update_layout(
                    xaxis_title="Moyenne /10",
                    yaxis_title="Matière",
                    yaxis=dict(autorange="reversed"),
                    height=500
                )
                st.plotly_chart(fig_matieres_faibles, width="stretch")

            with col_n2:
                st.subheader("📊 Notes faibles par matière")

                top_notes_faibles = matieres_filtrees.sort_values(
                    by="nombre_notes_faibles",
                    ascending=False
                ).head(10)

                fig_notes_faibles = px.bar(
                    top_notes_faibles,
                    x="nombre_notes_faibles",
                    y="matiere_principale",
                    orientation="h",
                    text="nombre_notes_faibles",
                    title="Nombre de notes faibles par matière"
                )
                fig_notes_faibles.update_traces(textposition="outside")
                fig_notes_faibles.update_layout(
                    xaxis_title="Nombre de notes faibles",
                    yaxis_title="Matière",
                    yaxis=dict(autorange="reversed"),
                    height=500
                )
                st.plotly_chart(fig_notes_faibles, width="stretch")

            st.subheader("Détail des matières")
            st.dataframe(
                matieres_filtrees[[
                    "classe",
                    "niveau",
                    "matiere_principale",
                    "composante",
                    "moyenne",
                    "nombre_notes",
                    "nombre_notes_faibles"
                ]],
                width="stretch"
            )

    # =========================
    # PAGE 5: ABSENCE ANALYSIS
    # =========================

    elif page == "Analyse des absences":
        st.header("📅 Analyse des absences")

        classes_abs = ["Toutes"] + sorted(synthese_eleves["classe"].dropna().unique().tolist())
        classe_abs = st.selectbox("Filtrer par classe", classes_abs)

        df_abs = synthese_eleves.copy()
        abs_mens = absences_mensuelles.copy()

        if classe_abs != "Toutes":
            df_abs = df_abs[df_abs["classe"] == classe_abs]
            abs_mens = abs_mens[abs_mens["classe"] == classe_abs]

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            top_absents = df_abs.sort_values(
                by="total_absences",
                ascending=False
            ).head(10)

            fig_absents = px.bar(
                top_absents,
                x="total_absences",
                y="nom_complet",
                orientation="h",
                text="total_absences",
                title="Top 10 élèves les plus absents"
            )
            fig_absents.update_traces(textposition="outside")
            fig_absents.update_layout(
                xaxis_title="Total absences",
                yaxis_title="Élève",
                yaxis=dict(autorange="reversed"),
                height=500
            )
            st.plotly_chart(fig_absents, width="stretch")

        with col_a2:
            if "numero_mois" in abs_mens.columns:
                mois_col = "numero_mois"
            elif "numro_mois" in abs_mens.columns:
                mois_col = "numro_mois"
            else:
                mois_col = "mois"

            resume_mois = abs_mens.groupby(["mois", mois_col]).agg(
                total_absences=("total_absences", "sum")
            ).reset_index().sort_values(mois_col)

            fig_mois = px.line(
                resume_mois,
                x="mois",
                y="total_absences",
                markers=True,
                text="total_absences",
                title="Évolution mensuelle des absences"
            )
            fig_mois.update_traces(textposition="top center")
            fig_mois.update_layout(
                xaxis_title="Mois",
                yaxis_title="Total absences",
                xaxis=dict(type="category"),
                height=500
            )
            st.plotly_chart(fig_mois, width="stretch")

        st.subheader("Détail des absences")
        st.dataframe(
            df_abs[[
                "id_eleve",
                "nom_complet",
                "classe",
                "moyenne_generale",
                "total_absences",
                "categorie_absences",
                "statut_final",
                "recommandation"
            ]],
            width="stretch"
        )

    # =========================
    # PAGE 6: WEAK STUDENTS
    # =========================

    elif page == "Élèves faibles":
        st.header("🚨 Élèves faibles et élèves à risque")

        col_w1, col_w2 = st.columns(2)

        with col_w1:
            st.subheader("Élèves faibles")
            faibles = synthese_eleves[
                synthese_eleves["statut_scolaire"].isin(["À risque", "Faible"])
            ].copy()

            st.write(f"Nombre d'élèves faibles : {len(faibles)}")
            st.dataframe(
                faibles[[
                    "id_eleve",
                    "nom_complet",
                    "classe",
                    "moyenne_generale",
                    "total_absences",
                    "statut_scolaire",
                    "statut_final",
                    "matieres_faibles",
                    "recommandation"
                ]],
                width="stretch"
            )

        with col_w2:
            st.subheader("Élèves à risque")
            risque = synthese_eleves[synthese_eleves["statut_final"] == "À risque"].copy()

            st.write(f"Nombre d'élèves à risque : {len(risque)}")

            if risque.empty:
                st.success("Aucun élève à risque selon les critères actuels.")
            else:
                st.dataframe(
                    risque[[
                        "id_eleve",
                        "nom_complet",
                        "classe",
                        "moyenne_generale",
                        "total_absences",
                        "statut_scolaire",
                        "statut_final",
                        "matieres_faibles",
                        "recommandation"
                    ]],
                    width="stretch"
                )
    # =========================
    # PAGE 7: RISK PREDICTION
    # =========================

    elif page == "Prédiction du risque":
        st.header("🤖 Prédiction du risque scolaire")

        st.info(
            "Ce module utilise un modèle de Machine Learning avec Scikit-learn "
            "pour estimer le niveau de risque des élèves à partir de leurs moyennes, "
            "absences et matières faibles."
        )

        model, data_prediction, accuracy, X_test, y_test = creer_modele_risque(synthese_eleves)

        if model is None:
            st.warning("Les données ne sont pas suffisantes pour entraîner le modèle.")
        else:
            col_ml1, col_ml2, col_ml3 = st.columns(3)

            with col_ml1:
                make_kpi("Modèle utilisé", "Decision Tree")

            with col_ml2:
                make_kpi("Précision", f"{accuracy * 100:.1f}%")

            with col_ml3:
                make_kpi("Élèves analysés", len(data_prediction))

            st.divider()

            st.subheader("📊 Répartition des risques prédits")

            risque_count = data_prediction["risque_predit"].value_counts().reset_index()
            risque_count.columns = ["risque_predit", "nombre_eleves"]

            fig_risque = px.pie(
                risque_count,
                names="risque_predit",
                values="nombre_eleves",
                title="Répartition des élèves par risque prédit"
            )

            st.plotly_chart(fig_risque, width="stretch")

            st.subheader("🚨 Élèves avec risque élevé prédit")

            eleves_risque_eleve = data_prediction[
                data_prediction["risque_predit"] == "Risque élevé"
            ].sort_values(
                by=["total_absences", "moyenne_generale"],
                ascending=[False, True]
            )

            if eleves_risque_eleve.empty:
                st.success("Aucun élève classé en risque élevé par le modèle.")
            else:
                st.dataframe(
                    eleves_risque_eleve[[
                        "id_eleve",
                        "nom_complet",
                        "classe",
                        "moyenne_generale",
                        "total_absences",
                        "nombre_matieres_faibles",
                        "statut_scolaire",
                        "statut_final",
                        "risque_predit",
                        "recommandation"
                    ]],
                    width="stretch"
                )

            st.divider()

            st.subheader("👤 Tester la prédiction pour un élève")

            classes_ml = ["Toutes"] + sorted(data_prediction["classe"].dropna().unique().tolist())
            classe_ml = st.selectbox("Filtrer par classe", classes_ml)

            df_ml = data_prediction.copy()

            if classe_ml != "Toutes":
                df_ml = df_ml[df_ml["classe"] == classe_ml]

            liste_eleves_ml = sorted(df_ml["nom_complet"].dropna().unique().tolist())

            if liste_eleves_ml:
                eleve_ml = st.selectbox("Sélectionner un élève", liste_eleves_ml)

                profil_ml = df_ml[df_ml["nom_complet"] == eleve_ml].iloc[0]

                col_e1, col_e2, col_e3, col_e4 = st.columns(4)

                with col_e1:
                    make_kpi("Moyenne", f"{profil_ml['moyenne_generale']:.2f}/10")

                with col_e2:
                    make_kpi("Absences", int(profil_ml["total_absences"]))

                with col_e3:
                    make_kpi("Matières faibles", int(profil_ml["nombre_matieres_faibles"]))

                with col_e4:
                    make_kpi("Risque prédit", profil_ml["risque_predit"])

                st.write("**Statut actuel :**", profil_ml["statut_final"])
                st.write("**Recommandation :**", profil_ml["recommandation"])

            st.divider()

            st.subheader("📌 Importance des critères")

            importance = pd.DataFrame({
                "critere": [
                    "Moyenne générale",
                    "Total absences",
                    "Nombre de matières faibles"
                ],
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)

            fig_importance = px.bar(
                importance,
                x="importance",
                y="critere",
                orientation="h",
                text="importance",
                title="Importance des critères dans la prédiction"
            )

            fig_importance.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )

            fig_importance.update_layout(
                xaxis_title="Importance",
                yaxis_title="Critère",
                yaxis=dict(autorange="reversed"),
                height=400
            )

            st.plotly_chart(fig_importance, width="stretch")

            st.caption(
                "Ce module est expérimental. Il sert d’aide à la décision et ne remplace pas "
                "l’analyse pédagogique de l’administration."
            )
    # =========================
    # PAGE 8: FOLLOW-UP
    # =========================

    elif page == "Suivi élèves":
        st.header("📝 Suivi des élèves")

        suivi = charger_suivi()

        st.subheader("Ajouter une action de suivi")

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            classe_suivi = st.selectbox(
                "Classe",
                sorted(synthese_eleves["classe"].dropna().unique().tolist())
            )

            eleves_classe = synthese_eleves[synthese_eleves["classe"] == classe_suivi]
            eleve_suivi = st.selectbox(
                "Élève",
                sorted(eleves_classe["nom_complet"].dropna().unique().tolist())
            )

            type_action = st.selectbox(
                "Type d'action",
                [
                    "Contact des parents",
                    "Soutien scolaire",
                    "Suivi des absences",
                    "Réunion avec l'élève",
                    "Observation",
                    "Autre"
                ]
            )

            responsable = st.text_input("Responsable", value="Administration")

        with col_s2:
            date_suivi = st.date_input("Date du suivi", value=date.today())
            statut_suivi = st.selectbox(
                "Statut du suivi",
                ["À faire", "En cours", "Terminé"]
            )
            prochaine_action = st.text_input("Prochaine action")

        observation = st.text_area("Observation")

        if st.button("Enregistrer le suivi"):
            eleve_row = eleves_classe[eleves_classe["nom_complet"] == eleve_suivi].iloc[0]

            ajouter_suivi(
                date_suivi=date_suivi,
                id_eleve=eleve_row["id_eleve"],
                nom_complet=eleve_row["nom_complet"],
                classe=eleve_row["classe"],
                type_action=type_action,
                observation=observation,
                responsable=responsable,
                prochaine_action=prochaine_action,
                statut_suivi=statut_suivi
            )

            st.success("✅ Action de suivi enregistrée avec succès dans la base de données.")

            suivi = charger_suivi()

        st.divider()

        st.subheader("Historique du suivi")

        if suivi.empty:
            st.info("Aucune action de suivi enregistrée pour le moment.")
        else:
            st.dataframe(suivi, width="stretch")

    # =========================
    # PAGE 9: DATA QUALITY
    # =========================

    elif page == "Qualité des données":
        st.header("✅ Qualité des données")

        col_q1, col_q2, col_q3 = st.columns(3)

        with col_q1:
            make_kpi("Élèves", len(etudiants))

        with col_q2:
            make_kpi("Lignes de notes", len(notes))

        with col_q3:
            make_kpi("Lignes d'absences", len(absences))

        st.divider()

        st.subheader("Colonnes disponibles")

        with st.expander("Synthèse élèves"):
            st.write(synthese_eleves.columns.tolist())

        with st.expander("Notes"):
            st.write(notes.columns.tolist())

        with st.expander("Absences"):
            st.write(absences.columns.tolist())

        st.subheader("Valeurs manquantes principales")

        missing_summary = pd.DataFrame({
            "table": ["Synthèse élèves", "Notes", "Absences"],
            "valeurs_manquantes": [
                int(synthese_eleves.isna().sum().sum()),
                int(notes.isna().sum().sum()),
                int(absences.isna().sum().sum())
            ]
        })

        st.dataframe(missing_summary, width="stretch")
    # =========================
    # PAGE 10: ABOUT PROJECT
    # =========================

    elif page == "À propos du projet":
        st.header("ℹ️ À propos du projet")

        st.markdown("""
        Cette application est un **prototype de suivi des performances scolaires** réalisé dans le cadre d’un stage.

        L’objectif principal est d’aider l’administration scolaire à mieux exploiter les données disponibles afin de suivre les élèves,
        analyser les notes et les absences, identifier les élèves faibles ou à risque, et enregistrer les actions de suivi.
        """)

        st.divider()

        st.subheader("🎯 Objectifs du projet")

        st.markdown("""
        - Centraliser les données scolaires dans une application interactive.
        - Visualiser les performances des élèves par classe, matière et statut.
        - Suivre les absences mensuelles et détecter les élèves les plus absents.
        - Identifier les élèves faibles ou à risque.
        - Enregistrer les actions de suivi administratif.
        - Ajouter un module expérimental de prédiction du risque scolaire.
        """)

        st.subheader("📌 Problématique")

        st.markdown("""
        Les données scolaires sont souvent disponibles sous forme de fichiers séparés exportés depuis la plateforme **Massar** :
        listes des élèves, notes, absences, classes, etc.

        Cette séparation rend l’analyse globale difficile pour l’administration, surtout lorsqu’il faut répondre rapidement à des questions comme :

        - Quels élèves ont besoin d’un suivi ?
        - Quelles classes ont les moyennes les plus faibles ?
        - Quelles matières posent le plus de difficultés ?
        - Quels élèves combinent faibles résultats et absences élevées ?
        """)

        st.subheader("📂 Source des données")

        st.markdown("""
        Les données utilisées dans cette version proviennent de fichiers Excel fournis par l’établissement scolaire.

        Ces données ont été nettoyées, structurées et regroupées dans un fichier central d’analyse utilisé par l’application.
        """)

        st.subheader("⚙️ Fonctionnement actuel")

        st.markdown("""
        Dans la version actuelle, l’application utilise un fichier Excel centralisé contenant les informations préparées :

        - élèves
        - notes
        - absences
        - synthèse par élève
        - synthèse par classe
        - synthèse par matière

        L’utilisateur peut actualiser les données après modification du fichier principal.
        """)

        st.subheader("🛠️ Technologies utilisées")

        technologies = pd.DataFrame({
            "Technologie": [
                "Python",
                "Pandas",
                "Plotly",
                "Streamlit",
                "CSS",
                "SQLite",
                "Scikit-learn",
                "Excel"
            ],
            "Utilisation dans le projet": [
                "Logique de l’application et traitement des données",
                "Nettoyage, manipulation et analyse des données",
                "Création de graphiques interactifs",
                "Création de l’interface web interactive",
                "Amélioration du design de l’interface",
                "Stockage des actions de suivi des élèves",
                "Module expérimental de prédiction du risque scolaire",
                "Source et stockage des données scolaires"
            ]
        })

        st.dataframe(technologies, use_container_width=True)

        st.subheader("🤖 Module de prédiction du risque")

        st.markdown("""
        Un module expérimental de Machine Learning a été ajouté avec **Scikit-learn**.

        Il utilise certains indicateurs comme :

        - la moyenne générale
        - le total des absences
        - le nombre de matières faibles

        pour estimer le niveau de risque scolaire d’un élève.

        Ce module est une **aide à la décision**. Il ne remplace pas l’analyse pédagogique de l’administration ou des enseignants.
        """)

        st.subheader("🗄️ Base de données SQLite")

        st.markdown("""
        Le module de suivi des élèves utilise une base de données **SQLite** pour enregistrer l’historique des actions administratives.

        Cela permet de conserver les observations, les actions réalisées, les responsables et les prochaines actions à effectuer.
        """)

        st.subheader("⚠️ Limites actuelles")

        st.markdown("""
        - L’application utilise actuellement un fichier Excel préparé manuellement.
        - L’importation automatique directe depuis Massar n’est pas encore intégrée.
        - Le module de prédiction reste expérimental car les données disponibles sont limitées.
        - La qualité des résultats dépend de la qualité des données fournies.
        """)

        st.subheader("🚀 Évolutions futures")

        st.markdown("""
        - Ajouter un module d’importation automatique des fichiers exportés depuis Massar.
        - Permettre à l’administration de charger directement les fichiers de notes et d’absences.
        - Automatiser le nettoyage et la fusion des données.
        - Ajouter des comptes utilisateurs avec différents rôles.
        - Améliorer le modèle de prédiction avec plus de données historiques.
        - Générer automatiquement des rapports PDF pour l’administration.
        """)

        st.success("Cette application représente une première version fonctionnelle et réutilisable pour le suivi scolaire.")
except Exception as e:
    st.error("❌ Erreur lors du chargement de l'application.")
    st.write(e)