import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const ModelExplanationDashboard = () => {
  // Dataset options
  const datasets = ["Dutch"];
  
  // Model options
  const models = ["BB_Dutch"];
  
  // Sample data
  const samples = [
    {
      id: "sample1",
      name: "Sample 1",
      data: {
        age: 9,
        household_position: 1131,
        household_size: 112,
        prev_residence_place: 1,
        citizenship: 1,
        country_birth: 1,
        edu_level: 4,
        economic_status: 111,
        cur_eco_activity: 122,
        Marital_status: 4,
        sex_binary: 0
      },
      prediction: {
        label: 1,
        probabilities: [0.16402084, 0.83597916]
      },
      featureImportance: [
        -0.02942215, 0.00779719, 0.00401248, 0.00339498, 0.0, 
        -0.00492565, 0.09259971, 0.03374485, -0.00093868, 0.00063837, 0.20458916
      ],
      decisionRules: [
        ['sex_binary <= 0.00', 0.2912951688869976],
        ['0.40 < edu_level <= 0.60', 0.12529023899016264],
        ['prev_residence_place <= 0.00', 0.12024077992982875],
        ['0.36 < age <= 0.55', -0.05164314144672377],
        ['country_birth <= 0.00', -0.04902579162237665],
        ['cur_eco_activity <= 0.71', -0.037580990003817046],
        ['household_position <= 0.10', 0.036629740900944376],
        ['household_size <= 0.07', 0.020277588558909286],
        ['economic_status <= 0.00', 0.011987086697775524],
        ['0.00 < Marital_status <= 0.33', -0.011966383625126196]
      ],
      similarSamples: [
        {
          similarity: 1.0,
          data: {
            age: 10,
            household_position: 1122,
            household_size: 113,
            prev_residence_place: 1,
            citizenship: 1,
            country_birth: 1,
            edu_level: 3,
            economic_status: 111,
            cur_eco_activity: 124,
            Marital_status: 2,
            sex_binary: 0,
            occupation_binary: 1
          },
          id: "48758"
        },
        {
          similarity: 1.0,
          data: {
            age: 10,
            household_position: 1121,
            household_size: 112,
            prev_residence_place: 1,
            citizenship: 1,
            country_birth: 1,
            edu_level: 2,
            economic_status: 111,
            cur_eco_activity: 124,
            Marital_status: 2,
            sex_binary: 0,
            occupation_binary: 1
          },
          id: "4249"
        },
        {
          similarity: 1.0,
          data: {
            age: 10,
            household_position: 1121,
            household_size: 112,
            prev_residence_place: 1,
            citizenship: 1,
            country_birth: 1,
            edu_level: 3,
            economic_status: 111,
            cur_eco_activity: 124,
            Marital_status: 2,
            sex_binary: 1,
            occupation_binary: 0
          },
          id: "N/A"
        }
      ]
    },
    {
      id: "sample2",
      name: "Sample 2",
      data: {
        age: 7,
        household_position: 1131,
        household_size: 112,
        prev_residence_place: 1,
        citizenship: 1,
        country_birth: 1,
        edu_level: 5,
        economic_status: 111,
        cur_eco_activity: 138,
        Marital_status: 1,
        sex_binary: 0
      },
      prediction: {
        label: 0,
        probabilities: [0.86464745, 0.13535258]
      },
      featureImportance: [
        -0.00782544, 0.00422991, -0.00202537, -0.00261117, 0.0, 
        0.00448547, 0.47456479, -0.01099995, 0.04268176, -0.01348994, -0.09987371
      ],
      decisionRules: [
        ['0.60 < edu_level <= 1.00', -0.4550781982046549],
        ['sex_binary <= 0.00', 0.27073169153941756],
        ['age <= 0.27', 0.09339496829436184],
        ['prev_residence_place <= 0.00', 0.07893404544743478],
        ['country_birth <= 0.00', -0.05718273197171243],
        ['citizenship <= 0.00', -0.041228587391163876],
        ['Marital_status <= 0.00', 0.028421249345087073],
        ['0.11 < household_position <= 0.19', -0.01062319708006088],
        ['cur_eco_activity > 0.93', -0.009474145521056973],
        ['household_size <= 0.07', 0.004768681924036221]
      ],
      similarSamples: [
        {
          similarity: 1.0,
          data: {
            age: 7,
            household_position: 1131,
            household_size: 112,
            prev_residence_place: 1,
            citizenship: 1,
            country_birth: 1,
            edu_level: 5,
            economic_status: 111,
            cur_eco_activity: 138,
            Marital_status: 1,
            sex_binary: 0,
            occupation_binary: 0
          },
          id: "51705"
        },
        {
          similarity: 1.0,
          data: {
            age: 7,
            household_position: 1131,
            household_size: 112,
            prev_residence_place: 1,
            citizenship: 1,
            country_birth: 1,
            edu_level: 5,
            economic_status: 111,
            cur_eco_activity: 138,
            Marital_status: 1,
            sex_binary: 0,
            occupation_binary: 0
          },
          id: "14594"
        },
        {
          similarity: 1.0,
          data: {
            age: 7,
            household_position: 1131,
            household_size: 112,
            prev_residence_place: 1,
            citizenship: 1,
            country_birth: 1,
            edu_level: 5,
            economic_status: 111,
            cur_eco_activity: 138,
            Marital_status: 1,
            sex_binary: 0,
            occupation_binary: 1
          },
          id: "21989"
        }
      ]
    }
  ];

  // State
  const [selectedDataset, setSelectedDataset] = useState(datasets[0]);
  const [selectedModel, setSelectedModel] = useState(models[0]);
  const [selectedSample, setSelectedSample] = useState(samples[0]);

  // Feature names for mapping to importance values
  const featureNames = [
    "age",
    "household_position",
    "household_size",
    "prev_residence_place",
    "citizenship",
    "country_birth",
    "edu_level",
    "economic_status",
    "cur_eco_activity",
    "Marital_status",
    "sex_binary"
  ];

  // Convert feature importance to chart data
  const getFeatureImportanceData = () => {
    return featureNames.map((feature, index) => ({
      feature,
      importance: selectedSample.featureImportance[index]
    }));
  };

  // Convert decision rules to chart data
  const getDecisionRulesData = () => {
    return selectedSample.decisionRules.map(rule => ({
      rule: rule[0],
      importance: rule[1]
    }));
  };

  // Create a simplified decision tree visualization
  const renderDecisionTree = () => {
    // Take top 3 rules for visualization
    const topRules = [...selectedSample.decisionRules]
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 3);
    
    return (
      <div className="flex flex-col items-center mt-6 relative">
        {/* Root node */}
        <div className="w-16 h-16 rounded-full bg-gray-100 border-2 border-gray-300 flex items-center justify-center mb-4">
          Root
        </div>
        
        {/* Connector line */}
        <div className="h-8 w-0.5 bg-gray-300 -mt-4 mb-4"></div>
        
        {/* First decision node */}
        <div className={`w-64 p-2 rounded-lg mb-4 text-center text-white ${
          topRules[0][1] >= 0 ? "bg-blue-500" : "bg-red-500"
        }`}>
          {topRules[0][0]}
        </div>
        
        {/* Branching */}
        <div className="flex justify-center w-full">
          <div className="flex flex-col items-center mx-4">
            <div className="h-8 w-0.5 bg-gray-300 mb-2"></div>
            <div className="text-sm text-gray-500 mb-2">True</div>
            <div className={`w-48 p-2 rounded-lg text-center text-white ${
              topRules[1][1] >= 0 ? "bg-blue-500" : "bg-red-500"
            }`}>
              {topRules[1][0]}
            </div>
          </div>
          
          <div className="flex flex-col items-center mx-4">
            <div className="h-8 w-0.5 bg-gray-300 mb-2"></div>
            <div className="text-sm text-gray-500 mb-2">False</div>
            <div className={`w-48 p-2 rounded-lg text-center text-white ${
              topRules[2][1] >= 0 ? "bg-blue-500" : "bg-red-500"
            }`}>
              {topRules[2][0]}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6 bg-gray-50">
      <h1 className="text-3xl font-bold text-center mb-6">Explainable AI Dashboard</h1>
      
      {/* Controls */}
      <div className="bg-white shadow-md rounded-lg p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Dataset</label>
            <select 
              className="w-full p-2 border border-gray-300 rounded-md bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
            >
              {datasets.map(dataset => (
                <option key={dataset} value={dataset}>{dataset}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Model</label>
            <select 
              className="w-full p-2 border border-gray-300 rounded-md bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Sample</label>
            <select 
              className="w-full p-2 border border-gray-300 rounded-md bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={selectedSample.id}
              onChange={(e) => setSelectedSample(samples.find(s => s.id === e.target.value))}
            >
              {samples.map(sample => (
                <option key={sample.id} value={sample.id}>{sample.name}</option>
              ))}
            </select>
          </div>
        </div>
      </div>
      
      {/* Sample Data and Prediction */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Sample Data Card */}
        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Sample Input Data</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-gray-100">
                  <th className="border p-2 text-left">Feature</th>
                  <th className="border p-2 text-left">Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(selectedSample.data).map(([key, value]) => (
                  <tr key={key} className="hover:bg-gray-50">
                    <td className="border p-2 font-medium">{key}</td>
                    <td className="border p-2">{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        {/* Prediction Card */}
        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Model Prediction</h2>
          <div className="flex items-center justify-center mb-6">
            <div className={`text-5xl font-bold rounded-full h-24 w-24 flex items-center justify-center ${
              selectedSample.prediction.label === 1 ? "bg-blue-100 text-blue-600" : "bg-red-100 text-red-600"
            }`}>
              {selectedSample.prediction.label}
            </div>
          </div>
          
          <h3 className="text-lg font-medium mb-2">Prediction Probabilities</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-100 p-4 rounded-md">
              <div className="text-sm text-gray-500">Class 0</div>
              <div className="text-lg font-semibold">{(selectedSample.prediction.probabilities[0] * 100).toFixed(2)}%</div>
              <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                <div 
                  className="bg-red-600 h-2.5 rounded-full" 
                  style={{ width: `${selectedSample.prediction.probabilities[0] * 100}%` }}
                ></div>
              </div>
            </div>
            
            <div className="bg-gray-100 p-4 rounded-md">
              <div className="text-sm text-gray-500">Class 1</div>
              <div className="text-lg font-semibold">{(selectedSample.prediction.probabilities[1] * 100).toFixed(2)}%</div>
              <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                <div 
                  className="bg-blue-600 h-2.5 rounded-full" 
                  style={{ width: `${selectedSample.prediction.probabilities[1] * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="bg-white shadow-md rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Feature Importance</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={getFeatureImportanceData()} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="feature" type="category" width={120} />
            <Tooltip />
            <Bar 
              dataKey="importance" 
              fill={(data) => data.importance >= 0 ? "#3b82f6" : "#ef4444"}
              label={{ position: 'right', formatter: (value) => value.toFixed(4) }}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      {/* Decision Rules */}
      <div className="bg-white shadow-md rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Decision Rules</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-medium mb-3">Decision Tree Visualization</h3>
            {renderDecisionTree()}
          </div>
          <div>
            <h3 className="text-lg font-medium mb-3">All Decision Rules</h3>
            <div className="overflow-y-auto max-h-96">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-gray-100">
                    <th className="border p-2 text-left">Rule</th>
                    <th className="border p-2 text-left">Importance</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedSample.decisionRules.map((rule, idx) => (
                    <tr key={idx} className="hover:bg-gray-50">
                      <td className="border p-2">{rule[0]}</td>
                      <td className={`border p-2 font-medium ${
                        rule[1] >= 0 ? "text-blue-600" : "text-red-600"
                      }`}>
                        {rule[1].toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      
      {/* Similar Samples */}
      <div className="bg-white shadow-md rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Exemplar-Based Explanation</h2>
        <p className="text-gray-700 mb-4">Similar samples from the training dataset that influence this prediction:</p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {selectedSample.similarSamples.map((sample, index) => (
            <div key={index} className="border rounded-lg p-4 bg-gray-50">
              <div className="mb-2">
                <span className="font-medium">Similar Sample {index + 1}</span>
              </div>
              <div className="text-xs text-gray-500 mb-1">ID: {sample.id}</div>
              <div className="overflow-y-auto max-h-48">
                <table className="w-full border-collapse text-sm">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border p-1 text-left">Feature</th>
                      <th className="border p-1 text-left">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(sample.data).map(([key, value]) => (
                      <tr key={key} className="hover:bg-white">
                        <td className="border p-1">{key}</td>
                        <td className="border p-1">{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelExplanationDashboard;
