import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Components
const Dashboard = () => {
  const [customers, setCustomers] = useState([]);
  const [stats, setStats] = useState(null);
  const [segments, setSegments] = useState([]);
  const [elbowData, setElbowData] = useState(null);
  const [clusterResult, setClusterResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('data');

  // Load initial data
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [customersRes, statsRes] = await Promise.all([
        axios.get(`${API}/data/customers`),
        axios.get(`${API}/data/stats`)
      ]);
      setCustomers(customersRes.data);
      setStats(statsRes.data);
    } catch (error) {
      console.error('Error loading data:', error);
    }
  };

  const generateSampleData = async () => {
    setLoading(true);
    try {
      await axios.post(`${API}/data/sample`);
      await loadData();
      alert('Sample data generated successfully!');
    } catch (error) {
      alert('Error generating sample data: ' + error.message);
    }
    setLoading(false);
  };

  const performElbowAnalysis = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/analysis/elbow`);
      setElbowData(response.data);
      setActiveTab('elbow');
    } catch (error) {
      alert('Error performing elbow analysis: ' + error.message);
    }
    setLoading(false);
  };

  const performClustering = async (algorithm, nClusters) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/analysis/cluster`, {
        algorithm: algorithm,
        n_clusters: nClusters,
        features: ['age', 'annual_income', 'spending_score']
      });
      setClusterResult(response.data);
      
      // Generate segments
      const segmentsRes = await axios.get(`${API}/analysis/segments`);
      setSegments(segmentsRes.data);
      
      await loadData(); // Reload customer data with cluster assignments
      setActiveTab('segments');
    } catch (error) {
      alert('Error performing clustering: ' + error.message);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Customer Segmentation & Targeting
          </h1>
          <p className="text-gray-600">
            Analyze customer behavior, segment audiences, and generate AI-powered business insights
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-md mb-8">
          <div className="flex border-b border-gray-200">
            {[
              { id: 'data', label: 'Data Overview' },
              { id: 'elbow', label: 'Elbow Analysis' },
              { id: 'clustering', label: 'Clustering' },
              { id: 'segments', label: 'Business Segments' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-3 font-medium ${
                  activeTab === tab.id
                    ? 'border-b-2 border-blue-500 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Data Overview Tab */}
        {activeTab === 'data' && (
          <div className="space-y-6">
            {/* Action Buttons */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-bold mb-4">Data Management</h2>
              <div className="flex gap-4">
                <button
                  onClick={generateSampleData}
                  disabled={loading}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                >
                  {loading ? 'Generating...' : 'Generate Sample Data'}
                </button>
              </div>
            </div>

            {/* Statistics */}
            {stats && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-800">Total Customers</h3>
                  <p className="text-3xl font-bold text-blue-600">{stats.total_customers}</p>
                </div>
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-800">Average Age</h3>
                  <p className="text-3xl font-bold text-green-600">{stats.age_stats?.mean?.toFixed(1)}</p>
                </div>
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-800">Average Income</h3>
                  <p className="text-3xl font-bold text-purple-600">
                    ${stats.income_stats?.mean?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                  </p>
                </div>
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-800">Average Spending</h3>
                  <p className="text-3xl font-bold text-orange-600">{stats.spending_stats?.mean?.toFixed(1)}</p>
                </div>
              </div>
            )}

            {/* Customer Data Table */}
            {customers.length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-bold mb-4">Customer Data Preview</h2>
                <div className="overflow-x-auto">
                  <table className="min-w-full table-auto">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left">Age</th>
                        <th className="px-4 py-2 text-left">Income</th>
                        <th className="px-4 py-2 text-left">Spending Score</th>
                        <th className="px-4 py-2 text-left">Gender</th>
                        <th className="px-4 py-2 text-left">Cluster</th>
                      </tr>
                    </thead>
                    <tbody>
                      {customers.slice(0, 10).map((customer, idx) => (
                        <tr key={idx} className="border-b">
                          <td className="px-4 py-2">{customer.age}</td>
                          <td className="px-4 py-2">${customer.annual_income?.toLocaleString()}</td>
                          <td className="px-4 py-2">{customer.spending_score}</td>
                          <td className="px-4 py-2">{customer.gender}</td>
                          <td className="px-4 py-2">
                            {customer.cluster_id !== undefined ? (
                              <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">
                                Cluster {customer.cluster_id}
                              </span>
                            ) : (
                              <span className="text-gray-400">Not assigned</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {customers.length > 10 && (
                    <p className="text-gray-500 text-sm mt-2">
                      Showing 10 of {customers.length} customers
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Elbow Analysis Tab */}
        {activeTab === 'elbow' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-bold mb-4">Elbow Analysis</h2>
              <p className="text-gray-600 mb-4">
                Find the optimal number of clusters using the elbow method and silhouette analysis.
              </p>
              <button
                onClick={performElbowAnalysis}
                disabled={loading || customers.length === 0}
                className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50"
              >
                {loading ? 'Analyzing...' : 'Perform Elbow Analysis'}
              </button>
            </div>

            {elbowData && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-bold mb-4">Inertia Values</h3>
                  <div className="space-y-2">
                    {elbowData.k_values.map((k, idx) => (
                      <div key={k} className="flex justify-between items-center">
                        <span>K = {k}</span>
                        <span className="font-mono text-sm">{elbowData.inertias[idx].toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-bold mb-4">Silhouette Scores</h3>
                  <div className="space-y-2">
                    {elbowData.k_values.map((k, idx) => (
                      <div key={k} className="flex justify-between items-center">
                        <span>K = {k}</span>
                        <span className="font-mono text-sm">{elbowData.silhouette_scores[idx].toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Clustering Tab */}
        {activeTab === 'clustering' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-bold mb-4">Clustering Analysis</h2>
              <p className="text-gray-600 mb-6">
                Apply clustering algorithms to segment your customers.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3">K-Means Clustering</h3>
                  <p className="text-gray-600 text-sm mb-4">
                    Fast and efficient for spherical clusters
                  </p>
                  <div className="space-y-2">
                    {[3, 4, 5, 6].map(k => (
                      <button
                        key={k}
                        onClick={() => performClustering('kmeans', k)}
                        disabled={loading || customers.length === 0}
                        className="w-full bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                      >
                        K-Means (K={k})
                      </button>
                    ))}
                  </div>
                </div>

                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3">Hierarchical Clustering</h3>
                  <p className="text-gray-600 text-sm mb-4">
                    Better for non-spherical clusters
                  </p>
                  <div className="space-y-2">
                    {[3, 4, 5, 6].map(k => (
                      <button
                        key={k}
                        onClick={() => performClustering('hierarchical', k)}
                        disabled={loading || customers.length === 0}
                        className="w-full bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 disabled:opacity-50"
                      >
                        Hierarchical (K={k})
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {clusterResult && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-xl font-bold mb-4">Clustering Results</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-gray-50 p-4 rounded">
                    <h4 className="font-semibold">Algorithm</h4>
                    <p className="text-lg">{clusterResult.algorithm}</p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded">
                    <h4 className="font-semibold">Clusters</h4>
                    <p className="text-lg">{clusterResult.n_clusters}</p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded">
                    <h4 className="font-semibold">Silhouette Score</h4>
                    <p className="text-lg">{clusterResult.silhouette_score.toFixed(3)}</p>
                  </div>
                </div>
                {clusterResult.inertia && (
                  <div className="bg-gray-50 p-4 rounded mb-4">
                    <h4 className="font-semibold">Inertia</h4>
                    <p className="text-lg">{clusterResult.inertia.toFixed(2)}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Business Segments Tab */}
        {activeTab === 'segments' && (
          <div className="space-y-6">
            {segments.length > 0 ? (
              segments.map((segment) => (
                <div key={segment.id} className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold text-gray-800">{segment.label}</h3>
                    <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                      Cluster {segment.cluster_id}
                    </span>
                  </div>
                  
                  <p className="text-gray-600 mb-6">{segment.description}</p>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-blue-600">{segment.customer_count}</p>
                      <p className="text-sm text-gray-500">Customers</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-green-600">{segment.avg_age.toFixed(1)}</p>
                      <p className="text-sm text-gray-500">Avg Age</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-purple-600">
                        ${segment.avg_income.toLocaleString(undefined, {maximumFractionDigits: 0})}
                      </p>
                      <p className="text-sm text-gray-500">Avg Income</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-orange-600">{segment.avg_spending.toFixed(1)}</p>
                      <p className="text-sm text-gray-500">Avg Spending</p>
                    </div>
                  </div>
                  
                  <div className="border-t pt-4">
                    <h4 className="font-semibold text-gray-800 mb-3">🤖 AI-Powered Business Recommendations</h4>
                    <ul className="space-y-2">
                      {segment.business_recommendations.map((rec, idx) => (
                        <li key={idx} className="flex items-start space-x-2">
                          <span className="text-green-500 mt-1">•</span>
                          <span className="text-gray-700">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              ))
            ) : (
              <div className="bg-white rounded-lg shadow-md p-6 text-center">
                <h3 className="text-xl font-bold text-gray-800 mb-2">No Segments Available</h3>
                <p className="text-gray-600 mb-4">
                  Please perform clustering analysis first to generate customer segments.
                </p>
                <button
                  onClick={() => setActiveTab('clustering')}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
                >
                  Go to Clustering
                </button>
              </div>
            )}
          </div>
        )}

        {/* Loading Overlay */}
        {loading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Processing...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <Dashboard />
    </div>
  );
}

export default App;