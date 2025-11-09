import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from collections import Counter
import tempfile

# Page config
st.set_page_config(
    page_title="TOP-250 Movies Explorer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for minimalistic design
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    h1 {
        color: #1f1f1f;
        font-weight: 300;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎬 TOP-250 Movies Explorer")
st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "Upload your TOP-250 movies file",
    type=['csv', 'json'],
    help="Upload a CSV or JSON file containing movie data"
)

# Load data function
@st.cache_data
def load_data(file, file_type):
    if file_type == 'csv':
        return pd.read_csv(file)
    else:
        data = json.load(file)
        return pd.DataFrame(data)

# Process data
if uploaded_file is not None:
    try:
        # Determine file type
        file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'json'
        
        # Load data
        df = load_data(uploaded_file, file_type)
        
        st.success(f"✅ Loaded {len(df)} movies successfully!")
        
        # Display raw data
        with st.expander("📊 View Raw Data Table"):
            st.dataframe(df, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⏱️ Patience",
            "🎥 Binge Watching Steven Spielberg",
            "👤 This is about me",
            "🏭 Workhorse",
            "💰 Cash Horse"
        ])
        
        # TAB 1: PATIENCE - Longest movies
        with tab1:
            st.header("⏱️ Patience: Longest Movies")
            st.markdown("*Movies that require the most patience to watch*")
            
            # Slider for interactivity
            top_n = st.slider(
                "Number of movies to display",
                min_value=5,
                max_value=20,
                value=10,
                key="patience_slider"
            )
            
            # Assuming there's a 'runtime' or 'duration' column
            runtime_col = None
            for col in ['runtime', 'duration', 'runtimeMinutes', 'length']:
                if col in df.columns:
                    runtime_col = col
                    break
            
            if runtime_col:
                longest = df.nlargest(top_n, runtime_col)
                
                # Display as table
                display_cols = [
                    col for col in ['title', 'name', runtime_col, 'year', 
                                   'director', 'genre', 'rating']
                    if col in df.columns
                ]
                st.dataframe(longest[display_cols], use_container_width=True)
                
                # Bar chart
                fig = px.bar(
                    longest,
                    x=runtime_col,
                    y='title' if 'title' in df.columns else 'name',
                    orientation='h',
                    title=f"Top {top_n} Longest Movies",
                    labels={runtime_col: "Runtime (minutes)"},
                    color=runtime_col,
                    color_continuous_scale="Blues"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No runtime/duration column found in the data")
        
        # TAB 2: BINGE WATCHING STEVEN SPIELBERG
        with tab2:
            st.header("🎥 Binge Watching Steven Spielberg")
            st.markdown("*All movies directed by Steven Spielberg*")
            
            # Find director column
            director_col = None
            for col in ['director', 'directors', 'directorName']:
                if col in df.columns:
                    director_col = col
                    break
            
            if director_col:
                spielberg = df[
                    df[director_col].str.contains(
                        'Spielberg', case=False, na=False
                    )
                ]
                
                if not spielberg.empty:
                    st.metric("Total Spielberg Movies", len(spielberg))
                    
                    # Display movies
                    display_cols = [
                        col for col in ['title', 'name', 'year', 'rating', 
                                       'genre', runtime_col]
                        if col in df.columns
                    ]
                    st.dataframe(spielberg[display_cols], use_container_width=True)
                    
                    # Timeline of Spielberg movies
                    if 'year' in df.columns:
                        year_counts = spielberg['year'].value_counts().sort_index()
                        fig = px.line(
                            x=year_counts.index,
                            y=year_counts.values,
                            title="Spielberg Movies Over Time",
                            labels={'x': 'Year', 'y': 'Number of Movies'},
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No Steven Spielberg movies found in this dataset")
            else:
                st.warning("No director column found in the data")
        
        # TAB 3: THIS IS ABOUT ME - Personal stats/recommendations
        with tab3:
            st.header("👤 This is about me")
            st.markdown("*Personalized movie statistics and insights*")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'year' in df.columns:
                    avg_year = df['year'].mean()
                    st.metric("Average Release Year", f"{avg_year:.0f}")
            
            with col2:
                if 'rating' in df.columns or 'imdbRating' in df.columns:
                    rating_col = 'rating' if 'rating' in df.columns else 'imdbRating'
                    avg_rating = df[rating_col].mean()
                    st.metric("Average Rating", f"{avg_rating:.1f}")
            
            with col3:
                if runtime_col:
                    total_runtime = df[runtime_col].sum()
                    st.metric(
                        "Total Watch Time",
                        f"{total_runtime/60:.0f} hours"
                    )
            
            # Genre analysis
            genre_col = None
            for col in ['genre', 'genres', 'genreList']:
                if col in df.columns:
                    genre_col = col
                    break
            
            if genre_col:
                st.subheader("🎨 Genre Distribution")
                
                # Extract all genres
                all_genres = []
                for genres in df[genre_col].dropna():
                    if isinstance(genres, str):
                        all_genres.extend([g.strip() for g in genres.split(',')])
                    elif isinstance(genres, list):
                        all_genres.extend(genres)
                
                genre_counts = Counter(all_genres)
                
                # Pie chart
                fig = px.pie(
                    values=list(genre_counts.values()),
                    names=list(genre_counts.keys()),
                    title="Genre Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 4: WORKHORSE - Most prolific directors/actors
        with tab4:
            st.header("🏭 Workhorse: Most Prolific Contributors")
            st.markdown("*Directors and actors with the most movies*")
            
            top_n_work = st.slider(
                "Number of contributors to display",
                min_value=5,
                max_value=20,
                value=10,
                key="workhorse_slider"
            )
            
            col1, col2 = st.columns(2)
            
            # Most prolific directors
            if director_col:
                with col1:
                    st.subheader("🎬 Top Directors")
                    director_counts = df[director_col].value_counts().head(top_n_work)
                    
                    fig = px.bar(
                        x=director_counts.values,
                        y=director_counts.index,
                        orientation='h',
                        title=f"Top {top_n_work} Directors",
                        labels={'x': 'Number of Movies', 'y': 'Director'},
                        color=director_counts.values,
                        color_continuous_scale="Viridis"
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Most prolific actors
            actor_col = None
            for col in ['stars', 'actors', 'cast', 'star']:
                if col in df.columns:
                    actor_col = col
                    break
            
            if actor_col:
                with col2:
                    st.subheader("⭐ Top Actors")
                    all_actors = []
                    for actors in df[actor_col].dropna():
                        if isinstance(actors, str):
                            all_actors.extend([a.strip() for a in actors.split(',')])
                        elif isinstance(actors, list):
                            all_actors.extend(actors)
                    
                    actor_counts = Counter(all_actors).most_common(top_n_work)
                    
                    fig = px.bar(
                        x=[count for _, count in actor_counts],
                        y=[actor for actor, _ in actor_counts],
                        orientation='h',
                        title=f"Top {top_n_work} Actors",
                        labels={'x': 'Number of Movies', 'y': 'Actor'},
                        color=[count for _, count in actor_counts],
                        color_continuous_scale="Plasma"
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
        
        # TAB 5: CASH HORSE - Highest grossing/rated movies
        with tab5:
            st.header("💰 Cash Horse: Top Performing Movies")
            st.markdown("*Movies with the highest ratings and commercial success*")
            
            top_n_cash = st.slider(
                "Number of movies to display",
                min_value=5,
                max_value=20,
                value=10,
                key="cash_slider"
            )
            
            # Find rating column
            rating_col = None
            for col in ['rating', 'imdbRating', 'score']:
                if col in df.columns:
                    rating_col = col
                    break
            
            if rating_col:
                top_rated = df.nlargest(top_n_cash, rating_col)
                
                display_cols = [
                    col for col in ['title', 'name', rating_col, 'year', 
                                   'director', 'genre']
                    if col in df.columns
                ]
                st.dataframe(top_rated[display_cols], use_container_width=True)
                
                # Scatter plot
                if 'year' in df.columns:
                    fig = px.scatter(
                        top_rated,
                        x='year',
                        y=rating_col,
                        size=rating_col,
                        hover_data=['title' if 'title' in df.columns else 'name'],
                        title=f"Top {top_n_cash} Rated Movies Over Time",
                        color=rating_col,
                        color_continuous_scale="YlOrRd"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No rating column found in the data")
        
        # WOW FEATURES SECTION
        st.markdown("---")
        st.header("✨ WOW Features")
        
        wow_tab1, wow_tab2, wow_tab3 = st.tabs([
            "📅 Timeline View",
            "☁️ Genre Word Cloud",
            "🕸️ Top Collaborations"
        ])
        
        # WOW 1: Timeline View
        with wow_tab1:
            st.subheader("📅 Movies Released Per Year")
            
            if 'year' in df.columns:
                year_counts = df['year'].value_counts().sort_index()
                
                fig = px.bar(
                    x=year_counts.index,
                    y=year_counts.values,
                    title="Number of Movies per Year",
                    labels={'x': 'Year', 'y': 'Number of Movies'},
                    color=year_counts.values,
                    color_continuous_scale="Turbo"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Most Productive Year", year_counts.idxmax())
                with col2:
                    st.metric("Movies That Year", year_counts.max())
                with col3:
                    st.metric("Year Range", f"{year_counts.index.min()} - {year_counts.index.max()}")
            else:
                st.warning("No year column found")
        
        # WOW 2: Genre Word Cloud
        with wow_tab2:
            st.subheader("☁️ Genre Word Cloud")
            
            if genre_col:
                # Extract all genres
                all_genres = []
                for genres in df[genre_col].dropna():
                    if isinstance(genres, str):
                        all_genres.extend([g.strip() for g in genres.split(',')])
                    elif isinstance(genres, list):
                        all_genres.extend(genres)
                
                genre_text = ' '.join(all_genres)
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis'
                ).generate(genre_text)
                
                # Display
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.warning("No genre column found")
        
        # WOW 3: Top Collaborations Network
        with wow_tab3:
            st.subheader("🕸️ Top 10 Actor-Director Collaborations")
            
            if director_col and actor_col:
                collaborations = []
                
                for _, row in df.iterrows():
                    director = row.get(director_col)
                    actors = row.get(actor_col)
                    
                    if pd.notna(director) and pd.notna(actors):
                        if isinstance(actors, str):
                            actor_list = [a.strip() for a in actors.split(',')]
                        elif isinstance(actors, list):
                            actor_list = actors
                        else:
                            continue
                        
                        for actor in actor_list:
                            collaborations.append((director, actor))
                
                # Count collaborations
                collab_counts = Counter(collaborations)
                top_collabs = collab_counts.most_common(10)
                
                # Create network graph
                G = nx.Graph()
                
                for (director, actor), count in top_collabs:
                    G.add_edge(director, actor, weight=count, title=f"{count} movies")
                
                # Create PyVis network
                net = Network(height="600px", width="100%", bgcolor="#ffffff")
                net.from_nx(G)
                
                # Customize
                net.set_options("""
                {
                  "nodes": {
                    "font": {"size": 16}
                  },
                  "edges": {
                    "color": {"inherit": true},
                    "smooth": false
                  },
                  "physics": {
                    "enabled": true,
                    "barnesHut": {
                      "gravitationalConstant": -8000,
                      "springLength": 200
                    }
                  }
                }
                """)
                
                # Save and display
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
                    net.save_graph(f.name)
                    with open(f.name, 'r') as hf:
                        html = hf.read()
                        components.html(html, height=600)
                
                # Show table
                st.markdown("**Top Collaborations:**")
                collab_df = pd.DataFrame(
                    [(d, a, c) for (d, a), c in top_collabs],
                    columns=['Director', 'Actor', 'Collaborations']
                )
                st.dataframe(collab_df, use_container_width=True)
            else:
                st.warning("Director or actor column not found")
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.info("Please ensure your file contains movie data with appropriate columns")
else:
    st.info("👆 Please upload a CSV or JSON file to get started")
    
    # Sample data structure
    with st.expander("ℹ️ Expected Data Format"):
        st.markdown("""
        Your file should contain columns like:
        - `title` or `name`: Movie title
        - `year`: Release year
        - `rating` or `imdbRating`: Movie rating
        - `runtime` or `duration`: Movie length in minutes
        - `director`: Director name
        - `genre` or `genreList`: Genres (comma-separated or list)
        - `stars` or `actors`: Actor names (comma-separated or list)
        """)
