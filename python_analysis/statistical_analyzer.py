"""
Statistical Analyzer Module for Genomic Pleiotropy Analysis

This module provides statistical tools for analyzing trait correlations,
gene-trait associations, and pleiotropy patterns.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from typing import Dict, List, Tuple, Optional, Union
import warnings


class StatisticalAnalyzer:
    """Main class for statistical analysis of genomic pleiotropy data."""
    
    def __init__(self, multiple_testing_method: str = 'fdr_bh'):
        """
        Initialize the analyzer.
        
        Args:
            multiple_testing_method: Method for multiple testing correction
                                   ('bonferroni', 'fdr_bh', 'fdr_by', etc.)
        """
        self.multiple_testing_method = multiple_testing_method
        
    def calculate_trait_correlations(self,
                                   trait_data: pd.DataFrame,
                                   method: str = 'pearson',
                                   min_samples: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate correlations between traits with p-values.
        
        Args:
            trait_data: DataFrame with traits as columns
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_samples: Minimum number of non-null samples required
            
        Returns:
            Tuple of (correlation_matrix, p_value_matrix)
        """
        n_traits = trait_data.shape[1]
        trait_names = trait_data.columns
        
        # Initialize matrices
        corr_matrix = pd.DataFrame(
            np.eye(n_traits), 
            index=trait_names, 
            columns=trait_names
        )
        p_matrix = pd.DataFrame(
            np.zeros((n_traits, n_traits)), 
            index=trait_names, 
            columns=trait_names
        )
        
        # Calculate pairwise correlations
        for i in range(n_traits):
            for j in range(i + 1, n_traits):
                # Get non-null pairs
                mask = trait_data.iloc[:, i].notna() & trait_data.iloc[:, j].notna()
                x = trait_data.iloc[:, i][mask]
                y = trait_data.iloc[:, j][mask]
                
                if len(x) >= min_samples:
                    if method == 'pearson':
                        corr, p_val = stats.pearsonr(x, y)
                    elif method == 'spearman':
                        corr, p_val = stats.spearmanr(x, y)
                    elif method == 'kendall':
                        corr, p_val = stats.kendalltau(x, y)
                    else:
                        raise ValueError(f"Unknown correlation method: {method}")
                    
                    corr_matrix.iloc[i, j] = corr
                    corr_matrix.iloc[j, i] = corr
                    p_matrix.iloc[i, j] = p_val
                    p_matrix.iloc[j, i] = p_val
                else:
                    corr_matrix.iloc[i, j] = np.nan
                    corr_matrix.iloc[j, i] = np.nan
                    p_matrix.iloc[i, j] = np.nan
                    p_matrix.iloc[j, i] = np.nan
        
        return corr_matrix, p_matrix
    
    def test_gene_trait_association(self,
                                  gene_expression: pd.Series,
                                  trait_values: pd.Series,
                                  test_type: str = 'auto') -> Dict[str, float]:
        """
        Test association between gene expression and trait values.
        
        Args:
            gene_expression: Gene expression values
            trait_values: Trait values
            test_type: Type of test ('auto', 'continuous', 'categorical')
            
        Returns:
            Dictionary with test results
        """
        # Remove missing values
        mask = gene_expression.notna() & trait_values.notna()
        gene_expr = gene_expression[mask]
        trait_vals = trait_values[mask]
        
        if len(gene_expr) < 3:
            return {'test': 'insufficient_data', 'p_value': np.nan}
        
        # Determine test type if auto
        if test_type == 'auto':
            # Check if trait is categorical (few unique values)
            n_unique = trait_vals.nunique()
            if n_unique <= 5 or n_unique < len(trait_vals) * 0.1:
                test_type = 'categorical'
            else:
                test_type = 'continuous'
        
        results = {}
        
        if test_type == 'continuous':
            # Correlation test
            corr, p_val = stats.pearsonr(gene_expr, trait_vals)
            results = {
                'test': 'pearson_correlation',
                'correlation': corr,
                'p_value': p_val,
                'n_samples': len(gene_expr)
            }
            
        elif test_type == 'categorical':
            # ANOVA or t-test
            unique_categories = trait_vals.unique()
            
            if len(unique_categories) == 2:
                # T-test
                group1 = gene_expr[trait_vals == unique_categories[0]]
                group2 = gene_expr[trait_vals == unique_categories[1]]
                t_stat, p_val = stats.ttest_ind(group1, group2)
                results = {
                    'test': 't_test',
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'group1_mean': group1.mean(),
                    'group2_mean': group2.mean(),
                    'n_samples': len(gene_expr)
                }
            else:
                # One-way ANOVA
                groups = [gene_expr[trait_vals == cat] for cat in unique_categories]
                f_stat, p_val = stats.f_oneway(*groups)
                results = {
                    'test': 'anova',
                    'f_statistic': f_stat,
                    'p_value': p_val,
                    'n_groups': len(unique_categories),
                    'n_samples': len(gene_expr)
                }
        
        return results
    
    def calculate_pleiotropy_score(self,
                                 gene_trait_associations: Dict[str, List[str]],
                                 trait_correlations: Optional[pd.DataFrame] = None,
                                 method: str = 'count_weighted') -> Dict[str, float]:
        """
        Calculate pleiotropy scores for genes.
        
        Args:
            gene_trait_associations: Dict mapping genes to associated traits
            trait_correlations: Optional correlation matrix between traits
            method: Scoring method ('count', 'count_weighted', 'entropy')
            
        Returns:
            Dictionary mapping genes to pleiotropy scores
        """
        scores = {}
        
        for gene, traits in gene_trait_associations.items():
            n_traits = len(traits)
            
            if method == 'count':
                # Simple count of associated traits
                scores[gene] = n_traits
                
            elif method == 'count_weighted' and trait_correlations is not None:
                # Weight by inverse of trait correlations
                if n_traits <= 1:
                    scores[gene] = n_traits
                else:
                    # Calculate average correlation between associated traits
                    trait_subset = [t for t in traits if t in trait_correlations.columns]
                    if len(trait_subset) > 1:
                        corr_subset = trait_correlations.loc[trait_subset, trait_subset]
                        # Get upper triangle (excluding diagonal)
                        upper_triangle = corr_subset.values[np.triu_indices_from(corr_subset.values, k=1)]
                        avg_corr = np.nanmean(np.abs(upper_triangle))
                        # Weight: higher score for less correlated traits
                        scores[gene] = n_traits * (1 - avg_corr)
                    else:
                        scores[gene] = n_traits
                        
            elif method == 'entropy':
                # Shannon entropy based on trait diversity
                if n_traits == 0:
                    scores[gene] = 0
                else:
                    # Assuming equal probability for each trait
                    p = 1.0 / n_traits
                    entropy = -n_traits * p * np.log2(p) if p > 0 else 0
                    scores[gene] = entropy
            
            else:
                raise ValueError(f"Unknown scoring method: {method}")
        
        return scores
    
    def perform_pca_analysis(self,
                           trait_data: pd.DataFrame,
                           n_components: Optional[int] = None,
                           standardize: bool = True) -> Dict[str, Union[pd.DataFrame, np.ndarray, List[float]]]:
        """
        Perform PCA on trait data.
        
        Args:
            trait_data: DataFrame with traits as columns
            n_components: Number of components (None for all)
            standardize: Whether to standardize data before PCA
            
        Returns:
            Dictionary with PCA results
        """
        # Remove rows with any missing values
        clean_data = trait_data.dropna()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clean_data)
        else:
            scaled_data = clean_data.values
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(scaled_data)
        
        # Create component DataFrame
        component_cols = [f'PC{i+1}' for i in range(transformed_data.shape[1])]
        components_df = pd.DataFrame(
            transformed_data,
            index=clean_data.index,
            columns=component_cols
        )
        
        # Create loadings DataFrame
        loadings_df = pd.DataFrame(
            pca.components_.T,
            index=trait_data.columns,
            columns=component_cols
        )
        
        results = {
            'transformed_data': components_df,
            'loadings': loadings_df,
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'n_samples': len(clean_data),
            'n_components': pca.n_components_
        }
        
        return results
    
    def cluster_traits(self,
                     trait_data: pd.DataFrame,
                     method: str = 'kmeans',
                     n_clusters: Optional[int] = None,
                     **kwargs) -> Dict[str, Union[np.ndarray, float]]:
        """
        Cluster traits based on their patterns.
        
        Args:
            trait_data: DataFrame with traits as columns
            method: Clustering method ('kmeans', 'dbscan')
            n_clusters: Number of clusters (for kmeans)
            **kwargs: Additional arguments for clustering algorithm
            
        Returns:
            Dictionary with clustering results
        """
        # Transpose to have traits as rows
        trait_matrix = trait_data.T.fillna(trait_data.T.mean(axis=1))
        
        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(trait_matrix)
        
        results = {}
        
        if method == 'kmeans':
            if n_clusters is None:
                # Use elbow method to find optimal k
                n_clusters = self._find_optimal_k(scaled_data, max_k=min(10, len(trait_matrix) - 1))
            
            kmeans = KMeans(n_clusters=n_clusters, **kwargs)
            labels = kmeans.fit_predict(scaled_data)
            
            results = {
                'labels': labels,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'n_clusters': n_clusters,
                'trait_clusters': {trait: int(label) 
                                 for trait, label in zip(trait_data.columns, labels)}
            }
            
        elif method == 'dbscan':
            dbscan = DBSCAN(**kwargs)
            labels = dbscan.fit_predict(scaled_data)
            
            results = {
                'labels': labels,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'n_noise': list(labels).count(-1),
                'trait_clusters': {trait: int(label) 
                                 for trait, label in zip(trait_data.columns, labels)}
            }
        
        return results
    
    def _find_optimal_k(self, data: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection: find point with maximum curvature
        if len(inertias) < 3:
            return 3
        
        # Calculate second derivative
        second_diff = np.diff(np.diff(inertias))
        elbow_idx = np.argmax(second_diff) + 2  # +2 because of double diff and 0-indexing
        
        return elbow_idx
    
    def test_enrichment(self,
                      gene_set: List[str],
                      background_genes: List[str],
                      pathway_genes: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Test for enrichment of gene set in pathways.
        
        Args:
            gene_set: List of genes to test
            background_genes: List of all possible genes
            pathway_genes: Dict mapping pathway names to gene lists
            
        Returns:
            DataFrame with enrichment results
        """
        results = []
        
        for pathway_name, pathway_gene_list in pathway_genes.items():
            # Create contingency table
            in_pathway_in_set = len(set(gene_set) & set(pathway_gene_list))
            in_pathway_not_set = len(set(pathway_gene_list) - set(gene_set))
            not_pathway_in_set = len(set(gene_set) - set(pathway_gene_list))
            not_pathway_not_set = len(set(background_genes) - set(gene_set) - set(pathway_gene_list))
            
            # Fisher's exact test
            contingency_table = [[in_pathway_in_set, in_pathway_not_set],
                               [not_pathway_in_set, not_pathway_not_set]]
            
            odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')
            
            results.append({
                'pathway': pathway_name,
                'n_genes_in_pathway': len(pathway_gene_list),
                'n_genes_in_set': len(gene_set),
                'n_overlap': in_pathway_in_set,
                'odds_ratio': odds_ratio,
                'p_value': p_value
            })
        
        results_df = pd.DataFrame(results)
        
        # Multiple testing correction
        if len(results_df) > 0:
            _, p_adjusted, _, _ = multipletests(
                results_df['p_value'], 
                method=self.multiple_testing_method
            )
            results_df['p_adjusted'] = p_adjusted
        
        return results_df.sort_values('p_value')
    
    def calculate_trait_heritability(self,
                                   trait_data: pd.DataFrame,
                                   kinship_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Estimate heritability for traits (simplified version).
        
        Args:
            trait_data: DataFrame with traits as columns
            kinship_matrix: Optional kinship/relatedness matrix
            
        Returns:
            Dictionary mapping traits to heritability estimates
        """
        heritability = {}
        
        for trait in trait_data.columns:
            trait_values = trait_data[trait].dropna()
            
            if kinship_matrix is not None:
                # Simplified heritability estimation
                # In practice, would use mixed models (e.g., GCTA, LDSC)
                warnings.warn("Using simplified heritability estimation. "
                            "For accurate estimates, use specialized tools.")
                
                # Variance components estimation would go here
                # This is a placeholder
                h2 = np.random.uniform(0.2, 0.8)  # Placeholder
            else:
                # Without kinship matrix, estimate from population variance
                # This is very simplified and for demonstration only
                total_var = trait_values.var()
                # Assume some genetic component (placeholder)
                h2 = min(0.5, total_var / (total_var + 1))
            
            heritability[trait] = h2
        
        return heritability