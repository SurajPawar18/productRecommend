import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(layout="wide")

# Load your dataset
@st.cache_data  # Caching for faster reloading
def load_data():
    try:
        data = pd.read_csv('amazon.csv')
        return data
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

def get_autocomplete_options(data):
    return data['product_name'].unique().tolist()

def collaborative_filtering_recommendations(data, selected_product_name, num_recommendations=3):
    try:
        # Compute TF-IDF features for product names
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['product_name'])

        # Compute similarity scores based on product names
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Get selected product details
        selected_product_index = data[data['product_name'] == selected_product_name].index[0]
        selected_product_row = cosine_sim[selected_product_index]

        # Sort indices based on similarity scores
        similar_product_indices = selected_product_row.argsort()[-num_recommendations - 1:-1][::-1]

        # Display recommended products
        recommendations = []
        for index in similar_product_indices:
            recommended_product = data.iloc[index]
            recommendations.append(recommended_product)

        return recommendations
    except Exception as e:
        st.error(f"An error occurred while generating recommendations: {e}")
        return None


def main():
    st.title('Product Recommendation System for E-commerce')

    # Load the dataset
    data = load_data()

    if data is not None:
        # Get auto-suggest options
        autocomplete_options = get_autocomplete_options(data)

        # Search bar for product
        selected_product_name = st.selectbox('Search for a product:', autocomplete_options, key='product_search')

        if selected_product_name:
            # Collaborative Filtering recommendations
            recommendations = collaborative_filtering_recommendations(data, selected_product_name)

            if recommendations is not None:
                # Display selected product details
                selected_product = data[data['product_name'] == selected_product_name].iloc[0]

                st.divider()

                # Layout for product details and images
                detail_col, space_col, image_col = st.columns([2.5, 0.5, 1])

                # Product details
                with detail_col:
                    st.header("Selected Product Details")
                    st.write(f"**Product Name:** {selected_product['product_name']}")
                    st.write(f"**Category:** {selected_product['category']}")
                    if 'actual_price' in selected_product:
                        st.write(f"**Actual Price:** {selected_product['actual_price']}")
                    st.write("**Rating:** ", selected_product['rating'])

                # Space column
                with space_col:
                    pass  # Empty space for layout

                # Product image
                with image_col:
                    st.header("Product Image")
                    if pd.isnull(selected_product['img_link']):
                        st.write("Image is not available. Click [here]({}) to view the product.".format(selected_product["product_link"]))
                    else:
                        st.markdown(
                            f'<a href="{selected_product["product_link"]}" target="_blank"><img src="{selected_product["img_link"]}" style="max-width:100%; height:auto;" title="{selected_product_name}"></a>',
                            unsafe_allow_html=True
                        )

                st.divider()
                st.header("Recommended Products")
                st.divider()

                for product in recommendations:
                    # Split the layout into two columns for product details and image
                    detail_col, space_col, image_col = st.columns([2.5, 0.5, 1])

                    # Product details
                    with detail_col:
                        st.subheader(product['product_name'])
                        st.write(f"**Category:** {product['category']}")
                        if 'actual_price' in product:
                            st.write(f"**Actual Price:** {product['actual_price']}")
                        st.write("**Rating:** ", product['rating'])

                    # Space column
                    with space_col:
                        pass  # Empty space for layout

                    # Product image
                    with image_col:
                        if pd.isnull(product['img_link']):
                            st.write("Image is not available. Click [here]({}) to view the product.".format(product["product_link"]))
                        else:
                            st.markdown(
                                f'<a href="{product["product_link"]}" target="_blank"><img src="{product["img_link"]}" style="max-width:100%; height:auto;" title="{product["product_name"]}"></a>',
                                unsafe_allow_html=True
                            )

                    st.divider()


if __name__ == '__main__':
    main()
