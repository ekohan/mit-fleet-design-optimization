"""
Tests for the clustering module.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import pytest

from src.clustering import (
    Cluster, Symbols, get_clustering_input, compute_composite_distance,
    get_clustering_model, generate_clusters_for_configurations,
    compute_demands, process_configuration, _generate_feasibility_mapping,
    _is_customer_feasible, estimate_num_initial_clusters, validate_cluster_coverage
)
from src.config.parameters import Parameters

class TestClusterClass(unittest.TestCase):
    """Tests for the Cluster dataclass."""
    
    def setUp(self):
        """Set up test data."""
        self.customers = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Latitude': [1.0, 2.0],
            'Longitude': [1.0, 2.0],
            'Frozen_Demand': [10, 20],
            'Chilled_Demand': [15, 10],
            'Dry_Demand': [5, 15]
        })
        
        self.config = pd.Series({
            'Config_ID': 1,
            'Frozen': 1,
            'Chilled': 1,
            'Dry': 1,
            'Capacity': 100
        })
        
        self.depot = {'latitude': 0.0, 'longitude': 0.0}
        self.goods = ['Frozen', 'Chilled', 'Dry']
        
    def test_cluster_creation(self):
        """Test creating a Cluster instance."""
        cluster = Cluster(
            cluster_id=1,
            config_id=1,
            customers=['C1', 'C2'],
            total_demand={'Frozen': 30, 'Chilled': 25, 'Dry': 20},
            centroid_latitude=1.5,
            centroid_longitude=1.5,
            goods_in_config=['Frozen', 'Chilled', 'Dry'],
            route_time=2.5,
            method='minibatch_kmeans'
        )
        
        self.assertEqual(cluster.cluster_id, 1)
        self.assertEqual(cluster.config_id, 1)
        self.assertEqual(cluster.customers, ['C1', 'C2'])
        self.assertEqual(cluster.total_demand, {'Frozen': 30, 'Chilled': 25, 'Dry': 20})
        self.assertEqual(cluster.centroid_latitude, 1.5)
        self.assertEqual(cluster.centroid_longitude, 1.5)
        self.assertEqual(cluster.goods_in_config, ['Frozen', 'Chilled', 'Dry'])
        self.assertEqual(cluster.route_time, 2.5)
        self.assertEqual(cluster.method, 'minibatch_kmeans')
        
    def test_to_dict_method(self):
        """Test the to_dict method of Cluster."""
        cluster = Cluster(
            cluster_id=1,
            config_id=1,
            customers=['C1', 'C2'],
            total_demand={'Frozen': 30, 'Chilled': 25, 'Dry': 20},
            centroid_latitude=1.5,
            centroid_longitude=1.5,
            goods_in_config=['Frozen', 'Chilled', 'Dry'],
            route_time=2.5,
            method='minibatch_kmeans'
        )
        
        expected_dict = {
            'Cluster_ID': 1,
            'Config_ID': 1,
            'Customers': ['C1', 'C2'],
            'Total_Demand': {'Frozen': 30, 'Chilled': 25, 'Dry': 20},
            'Centroid_Latitude': 1.5,
            'Centroid_Longitude': 1.5,
            'Goods_In_Config': ['Frozen', 'Chilled', 'Dry'],
            'Route_Time': 2.5,
            'Method': 'minibatch_kmeans'
        }
        
        self.assertEqual(cluster.to_dict(), expected_dict)
    
    @patch('src.clustering.estimate_route_time')
    def test_from_customers_method(self, mock_estimate_route_time):
        """Test the from_customers class method of Cluster."""
        mock_estimate_route_time.return_value = 2.5
        
        cluster = Cluster.from_customers(
            customers=self.customers,
            config=self.config,
            cluster_id=1,
            goods=self.goods,
            depot=self.depot,
            service_time=10.0,
            avg_speed=50.0,
            route_time_estimation='Legacy',
            method='minibatch_kmeans'
        )
        
        self.assertEqual(cluster.cluster_id, 1)
        self.assertEqual(cluster.config_id, 1)
        self.assertEqual(set(cluster.customers), {'C1', 'C2'})
        self.assertEqual(cluster.total_demand, {'Frozen': 30.0, 'Chilled': 25.0, 'Dry': 20.0})
        self.assertEqual(cluster.centroid_latitude, 1.5)
        self.assertEqual(cluster.centroid_longitude, 1.5)
        self.assertEqual(set(cluster.goods_in_config), {'Frozen', 'Chilled', 'Dry'})
        self.assertEqual(cluster.route_time, 2.5)
        self.assertEqual(cluster.method, 'minibatch_kmeans')


class TestClusteringFunctions(unittest.TestCase):
    """Tests for clustering utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.customers = pd.DataFrame({
            'Customer_ID': ['C1', 'C2', 'C3'],
            'Latitude': [1.0, 2.0, 3.0],
            'Longitude': [1.0, 2.0, 3.0],
            'Frozen_Demand': [10, 20, 15],
            'Chilled_Demand': [15, 10, 0],
            'Dry_Demand': [5, 15, 10]
        })
        
        self.goods = ['Frozen', 'Chilled', 'Dry']
        
    def test_get_clustering_input_feature_based(self):
        """Test get_clustering_input for feature-based methods."""
        input_data = get_clustering_input(
            self.customers, 
            self.goods, 
            'minibatch_kmeans',
            0.8,
            0.2,
            'euclidean'
        )
        
        expected = self.customers[['Latitude', 'Longitude']].values
        np.testing.assert_array_equal(input_data, expected)
        
    def test_get_clustering_input_precomputed(self):
        """Test get_clustering_input for methods needing precomputed distances."""
        input_data = get_clustering_input(
            self.customers, 
            self.goods, 
            'agglomerative_geo_0.8_demand_0.2',
            0.8,
            0.2,
            'euclidean'
        )
        
        # Should be a distance matrix of shape (n_customers, n_customers)
        self.assertEqual(input_data.shape, (3, 3))
        
    def test_compute_composite_distance(self):
        """Test compute_composite_distance function."""
        distance_matrix = compute_composite_distance(
            self.customers,
            self.goods,
            0.8,
            0.2
        )
        
        # Should be a square matrix of shape (n_customers, n_customers)
        self.assertEqual(distance_matrix.shape, (3, 3))
        # Diagonal should be zeros (distance to self)
        self.assertTrue(np.allclose(np.diag(distance_matrix), np.zeros(3)))
        # Matrix should be symmetric
        self.assertTrue(np.allclose(distance_matrix, distance_matrix.T))
        
    def test_get_clustering_model(self):
        """Test get_clustering_model function for different methods."""
        # Test MiniBatchKMeans
        model = get_clustering_model(3, 'minibatch_kmeans')
        self.assertEqual(model.n_clusters, 3)
        
        # Test KMedoids
        model = get_clustering_model(3, 'kmedoids')
        self.assertEqual(model.n_clusters, 3)
        
        # Test AgglomerativeClustering
        model = get_clustering_model(3, 'agglomerative_geo_0.8_demand_0.2')
        self.assertEqual(model.n_clusters, 3)
        self.assertEqual(model.metric, 'precomputed')
        
        # Test GaussianMixture
        model = get_clustering_model(3, 'gaussian_mixture')
        self.assertEqual(model.n_components, 3)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            get_clustering_model(3, 'invalid_method')
    
    def test_compute_demands(self):
        """Test compute_demands function."""
        demands = compute_demands(self.customers, self.goods)
        
        # Check keys
        self.assertIn('total', demands)
        self.assertIn('by_good', demands)
        self.assertIn('weighted', demands)
        
        # Check shapes
        self.assertEqual(len(demands['total']), 3)  # Number of customers
        self.assertEqual(demands['by_good'].shape, (3, 3))  # (n_customers, n_goods)
        self.assertEqual(demands['weighted'].shape, (3, 3))  # (n_customers, n_goods)
        
        # Check total demand calculations
        expected_totals = [30, 45, 25]  # Sum of demands for each customer
        pd.testing.assert_series_equal(
            demands['total'],
            pd.Series(expected_totals, index=self.customers.index)
        )


class TestCustomerFeasibilityFunctions(unittest.TestCase):
    """Tests for customer feasibility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.customers = pd.DataFrame({
            'Customer_ID': ['C1', 'C2', 'C3'],
            'Latitude': [1.0, 2.0, 3.0],
            'Longitude': [1.0, 2.0, 3.0],
            'Frozen_Demand': [10, 20, 15],
            'Chilled_Demand': [15, 10, 0],
            'Dry_Demand': [5, 15, 10]
        })
        
        self.configurations = pd.DataFrame({
            'Config_ID': [1, 2, 3],
            'Frozen': [1, 0, 1],
            'Chilled': [1, 1, 0],
            'Dry': [1, 1, 1],
            'Capacity': [100, 50, 30]
        })
        
        self.goods = ['Frozen', 'Chilled', 'Dry']
        
    def test_is_customer_feasible(self):
        """Test _is_customer_feasible function."""
        # Customer with all demands, config with all goods, high capacity
        self.assertTrue(_is_customer_feasible(
            self.customers.iloc[0],
            self.configurations.iloc[0],
            self.goods
        ))
        
        # Customer with frozen demand, config without frozen
        self.assertFalse(_is_customer_feasible(
            self.customers.iloc[0],  # has frozen demand
            self.configurations.iloc[1],  # config without frozen
            self.goods
        ))
        
        # Customer with demand exceeding capacity
        self.assertFalse(_is_customer_feasible(
            self.customers.iloc[1],  # has 20 frozen demand
            self.configurations.iloc[2],  # capacity 30, but has frozen=1
            self.goods
        ))
        
    def test_generate_feasibility_mapping(self):
        """Test _generate_feasibility_mapping function."""
        mapping = _generate_feasibility_mapping(
            self.customers,
            self.configurations,
            self.goods
        )
        
        # C1 should be feasible for config 1 only
        self.assertEqual(mapping['C1'], [1])
        
        # C2 should be feasible for config 1 only due to frozen demand
        self.assertEqual(mapping['C2'], [1])
        
        # C3 should be feasible for configs 1 and 2 (no chilled demand)
        self.assertEqual(set(mapping['C3']), {1, 2})


class TestClusteringProcess(unittest.TestCase):
    """Tests for the clustering process functions."""
    
    def setUp(self):
        """Set up test data."""
        self.customers = pd.DataFrame({
            'Customer_ID': ['C1', 'C2', 'C3', 'C4'],
            'Latitude': [1.0, 1.1, 5.0, 5.1],
            'Longitude': [1.0, 1.1, 5.0, 5.1],
            'Frozen_Demand': [10, 10, 15, 15],
            'Chilled_Demand': [15, 15, 0, 0],
            'Dry_Demand': [5, 5, 10, 10]
        })
        
        self.configurations = pd.DataFrame({
            'Config_ID': [1, 2],
            'Frozen': [1, 0],
            'Chilled': [1, 1],
            'Dry': [1, 1],
            'Capacity': [100, 50]
        })
        
        self.goods = ['Frozen', 'Chilled', 'Dry']
        self.depot = {'latitude': 0.0, 'longitude': 0.0}
        
    @patch('src.clustering.estimate_route_time')
    def test_estimate_num_initial_clusters(self, mock_estimate_route_time):
        """Test estimate_num_initial_clusters function."""
        mock_estimate_route_time.return_value = 2.0
        
        num_clusters = estimate_num_initial_clusters(
            self.customers,
            self.configurations.iloc[0],  # Config with capacity 100
            self.depot,
            avg_speed=50.0,
            service_time=10.0,
            goods=self.goods,
            max_route_time=8.0,
            route_time_estimation='Legacy'
        )
        
        # With total demand of 110 and capacity 100, should need at least 2 clusters
        self.assertGreaterEqual(num_clusters, 2)
        
    @patch('src.clustering.estimate_route_time')
    @patch('src.clustering.get_clustering_model')
    def test_process_configuration(self, mock_get_model, mock_estimate_route_time):
        """Test process_configuration function."""
        # Mock route time to be below max
        mock_estimate_route_time.return_value = 2.0
        
        # Mock clustering model
        mock_model = MagicMock()
        mock_model.fit_predict.return_value = np.array([0, 0, 1, 1])
        mock_get_model.return_value = mock_model
        
        # Create feasibility mapping - all customers feasible for config 1
        feasible_customers = {
            'C1': [1], 'C2': [1], 'C3': [1], 'C4': [1]
        }
        
        clusters = process_configuration(
            config=self.configurations.iloc[0],
            customers=self.customers,
            goods=self.goods,
            depot=self.depot,
            avg_speed=50.0,
            service_time=10.0,
            max_route_time=8.0,
            feasible_customers=feasible_customers,
            max_split_depth=2,
            clustering_method='minibatch_kmeans',
            route_time_estimation='Legacy',
            geo_weight=0.8,
            demand_weight=0.2,
            distance_metric='euclidean'
        )
        
        # Should generate 2 clusters based on our mocked model
        self.assertEqual(len(clusters), 2)
        
        # Clusters should contain the correct customers
        cluster1_customers = set(clusters[0]['Customers'])
        cluster2_customers = set(clusters[1]['Customers'])
        self.assertEqual(cluster1_customers.union(cluster2_customers), 
                         set(['C1', 'C2', 'C3', 'C4']))
        
    def test_validate_cluster_coverage(self):
        """Test validate_cluster_coverage function."""
        clusters_df = pd.DataFrame({
            'Customers': [['C1', 'C2'], ['C3']]
        })
        
        # C4 not covered by any cluster
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1', 'C2', 'C3', 'C4']
        })
        
        # Function doesn't return anything, just logs uncovered customers
        validate_cluster_coverage(clusters_df, customers_df)


@pytest.fixture
def mock_parameters():
    """Create a mock Parameters object for testing."""
    params = MagicMock(spec=Parameters)
    params.goods = ['Frozen', 'Chilled', 'Dry']
    params.depot = {'latitude': 0.0, 'longitude': 0.0}
    params.avg_speed = 50.0
    params.service_time = 10.0
    params.max_route_time = 8.0
    params.clustering = {
        'method': 'minibatch_kmeans',
        'max_depth': 2,
        'route_time_estimation': 'Legacy',
        'geo_weight': 0.8,
        'demand_weight': 0.2,
        'distance': 'euclidean'
    }
    return params


@pytest.mark.parametrize("method", [
    'minibatch_kmeans',
    'kmedoids',
    'gaussian_mixture',
    'agglomerative_geo_0.8_demand_0.2',
    'combine'
])
@patch('src.clustering.process_configuration')
@patch('src.clustering._generate_feasibility_mapping')
def test_generate_clusters_for_configurations(mock_generate_mapping, mock_process_config, 
                                             method, mock_parameters):
    """Test generate_clusters_for_configurations with different methods."""
    # Set up test data
    customers = pd.DataFrame({
        'Customer_ID': ['C1', 'C2', 'C3', 'C4'],
        'Latitude': [1.0, 1.1, 5.0, 5.1],
        'Longitude': [1.0, 1.1, 5.0, 5.1],
        'Frozen_Demand': [10, 10, 15, 15],
        'Chilled_Demand': [15, 15, 0, 0],
        'Dry_Demand': [5, 5, 10, 10]
    })
    
    configurations = pd.DataFrame({
        'Config_ID': [1, 2],
        'Frozen': [1, 0],
        'Chilled': [1, 1],
        'Dry': [1, 1],
        'Capacity': [100, 50]
    })
    
    # Mock feasibility mapping
    mock_generate_mapping.return_value = {
        'C1': [1], 'C2': [1], 'C3': [1, 2], 'C4': [1, 2]
    }
    
    # Mock process_configuration to return some clusters
    mock_process_config.return_value = [
        {'Cluster_ID': 1001, 'Config_ID': 1, 'Customers': ['C1', 'C2'], 
         'Method': method},
        {'Cluster_ID': 1002, 'Config_ID': 1, 'Customers': ['C3', 'C4'], 
         'Method': method}
    ]
    
    # Update method in parameters
    mock_parameters.clustering['method'] = method
    
    # Call the function
    clusters_df = generate_clusters_for_configurations(
        customers,
        configurations,
        mock_parameters
    )
    
    # For 'combine' method, should call process_configuration with multiple methods
    if method == 'combine':
        assert mock_process_config.call_count > 2  # More than 2 calls for multiple methods
    else:
        # For single methods, should call process_configuration for each config (2)
        assert mock_process_config.call_count == 2
    
    # Should return a DataFrame with the correct clusters
    assert isinstance(clusters_df, pd.DataFrame)
    assert len(clusters_df) > 0 