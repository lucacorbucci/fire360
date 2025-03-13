import { useState } from "react";
import {
	BarChart,
	Bar,
	XAxis,
	YAxis,
	CartesianGrid,
	Tooltip,
	ResponsiveContainer,
} from "recharts";

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
			accuracy: 0.83,
			f1Score: 0.83,
			fidelity_dt: 1.0,
			localFidelity_dt: 1.0,
			fidelity_lr: 1.0,
			localFidelity_lr: 0.995,
			fidelity_knn: 1,
			localFidelity_knn: 1.0,
			localFidelity: 1.0,
			stabilityScore: 0.89,
			data: {
				age: 5,
				household_position: 1110,
				household_size: 112,
				prev_residence_place: 1,
				citizenship: 1,
				country_birth: 1,
				edu_level: 2,
				economic_status: 111,
				cur_eco_activity: 131,
				Marital_status: 1,
				sex_binary: 1,
			},
			prediction: {
				label: 1,
				probabilities: [0.16402084, 0.83597916],
			},
			featureImportance: [
				-5.314648352877113, 0.007978501769350919, 0.12280868684771122,
				-2.4425235761689956, 0.0, 0.0, -1.6403407080407693,
				0.12168432610281686, 0.1039957805705147, -7.647274793987985,
				0.0,
			],
			decisionRules: [
				["household_position <= 1115.5", 1110],
				["prev_residence_place <= 1.5", 1],
			],
			similarSamples: [
				{
					similarity: 1.0,
					data: {
						age: 8,
						household_position: 1122,
						household_size: 114,
						prev_residence_place: 1,
						citizenship: 1,
						country_birth: 1,
						edu_level: 3,
						economic_status: 111,
						cur_eco_activity: 132,
						Marital_status: 2,
						sex_binary: 0,
						occupation_binary: 1,
					},
					id: "25",
				},
				{
					similarity: 1.0,
					data: {
						age: 11,
						household_position: 1121,
						household_size: 112,
						prev_residence_place: 1,
						citizenship: 1,
						country_birth: 1,
						edu_level: 3,
						economic_status: 111,
						cur_eco_activity: 135,
						Marital_status: 2,
						sex_binary: 0,
						occupation_binary: 1,
					},
					id: "134",
				},
				{
					similarity: 1.0,
					data: {
						age: 12,
						household_position: 1121,
						household_size: 112,
						prev_residence_place: 1,
						citizenship: 1,
						country_birth: 1,
						edu_level: 2,
						economic_status: 111,
						cur_eco_activity: 131,
						Marital_status: 2,
						sex_binary: 0,
						occupation_binary: 0,
					},
					id: "59",
				},
			],
		},
		{
			id: "sample2",
			name: "Sample 2",
			accuracy: 0.83,
			f1Score: 0.83,
			fidelity_dt: 0.93,
			localFidelity_dt: 0.91,
			fidelity_lr: 1.0,
			localFidelity_lr: 0.985,
			fidelity_knn: 0,
			localFidelity_knn: 0.985,
			stabilityScore: 0.89,
			data: {
				age: 7,
				household_position: 1121,
				household_size: 112,
				prev_residence_place: 1,
				citizenship: 1,
				country_birth: 1,
				edu_level: 3,
				economic_status: 111,
				cur_eco_activity: 138,
				Marital_status: 2,
				sex_binary: 0,
			},
			prediction: {
				label: 0,
				probabilities: [0.86464745, 0.13535258],
			},
			featureImportance: [
				-2.1174852706643996, 0.026019895528212014, -0.22706379870548402,
				-4.146841309174532, 0.0, 1.007459994743197, -7.452820665539312,
				0.28173161488081955, 0.14779436163839021, -2.074674831347488,
				-12.94327266850943,
			],
			decisionRules: [
				["edu_level <= 4.5", 3],
				["sex_binary <= 0.5", 0],
				["prev_residence_place <= 1.5", 1],
				["household_size > 111.5", 112],
			],
			similarSamples: [
				{
					similarity: 1.0,
					data: {
						age: 8,
						household_position: 1122,
						household_size: 114,
						prev_residence_place: 1,
						citizenship: 1,
						country_birth: 1,
						edu_level: 3,
						economic_status: 111,
						cur_eco_activity: 132,
						Marital_status: 2,
						sex_binary: 0,
						occupation_binary: 1,
					},
					id: "51705",
				},
				{
					similarity: 1.0,
					data: {
						age: 11,
						household_position: 1121,
						household_size: 112,
						prev_residence_place: 1,
						citizenship: 1,
						country_birth: 1,
						edu_level: 5,
						economic_status: 111,
						cur_eco_activity: 138,
						Marital_status: 1,
						sex_binary: 0,
						occupation_binary: 0,
					},
					id: "14594",
				},
				{
					similarity: 1.0,
					data: {
						age: 12,
						household_position: 1121,
						household_size: 112,
						prev_residence_place: 1,
						citizenship: 1,
						country_birth: 1,
						edu_level: 2,
						economic_status: 111,
						cur_eco_activity: 131,
						Marital_status: 2,
						sex_binary: 0,
						occupation_binary: 1,
					},
					id: "21989",
				},
			],
		},
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
		"sex_binary",
	];

	// Convert feature importance to chart data
	const getFeatureImportanceData = () => {
		return featureNames.map((feature, index) => ({
			feature,
			importance: selectedSample.featureImportance[index],
		}));
	};

	// // Create a full decision tree visualization
	// const renderEnhancedDecisionTree = () => {
	// 	const rules = [...selectedSample.decisionRules].sort(
	// 		(a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1])),
	// 	);

	// 	// Flag for the prediction path highlighting
	// 	const isPredictionPath = (index: number) => index < 3; // Highlight first 3 nodes as the path to prediction

	// 	return (
	// 		<div className="flex flex-col items-center mt-6 relative overflow-x-auto w-full">
	// 			<div className="w-full overflow-x-auto pb-6">
	// 				<div className="flex flex-col items-center min-w-max">
	// 					{/* Root node */}
	// 					<div className="w-20 h-20 rounded-full bg-gray-100 border-2 border-gray-300 flex items-center justify-center mb-4 text-sm font-medium">
	// 						Root
	// 					</div>

	// 					{/* First level */}
	// 					<div className="h-10 w-0.5 bg-gray-300 -mt-4 mb-2"></div>
	// 					<div
	// 						className={`w-64 p-3 rounded-lg mb-6 text-center text-white ${
	// 							isPredictionPath(0)
	// 								? "ring-4 ring-yellow-500"
	// 								: ""
	// 						} ${
	// 							Number(rules[0][1]) >= 0
	// 								? "bg-blue-500"
	// 								: "bg-red-500"
	// 						}`}
	// 					>
	// 						{rules[0][0]}
	// 						<div className="text-xs mt-1 font-semibold">
	// 							Importance: {Number(rules[0][1]).toFixed(4)}
	// 						</div>
	// 					</div>

	// 					{/* Second level */}
	// 					<div className="flex justify-center w-full mb-6">
	// 						<div className="flex flex-col items-center mx-8">
	// 							<div className="h-10 w-0.5 bg-gray-300 mb-2"></div>
	// 							<div className="text-sm text-gray-500 mb-2">
	// 								True
	// 							</div>
	// 							<div
	// 								className={`w-60 p-3 rounded-lg text-center text-white ${
	// 									isPredictionPath(1)
	// 										? "ring-4 ring-yellow-500"
	// 										: ""
	// 								} ${
	// 									Number(rules[1][1]) >= 0
	// 										? "bg-blue-500"
	// 										: "bg-red-500"
	// 								}`}
	// 							>
	// 								{rules[1][0]}
	// 								<div className="text-xs mt-1 font-semibold">
	// 									Importance:{" "}
	// 									{Number(rules[1][1]).toFixed(4)}
	// 								</div>
	// 							</div>
	// 						</div>

	// 						<div className="flex flex-col items-center mx-8">
	// 							<div className="h-10 w-0.5 bg-gray-300 mb-2"></div>
	// 							<div className="text-sm text-gray-500 mb-2">
	// 								False
	// 							</div>
	// 							<div
	// 								className={`w-60 p-3 rounded-lg text-center text-white ${
	// 									Number(rules[2][1]) >= 0
	// 										? "bg-blue-500"
	// 										: "bg-red-500"
	// 								}`}
	// 							>
	// 								{rules[2][0]}
	// 								<div className="text-xs mt-1 font-semibold">
	// 									Importance:{" "}
	// 									{Number(rules[2][1]).toFixed(4)}
	// 								</div>
	// 							</div>
	// 						</div>
	// 					</div>

	// 					{/* Third level */}
	// 					<div className="flex justify-center w-full space-x-4 mb-6">
	// 						<div className="flex flex-col items-center">
	// 							<div className="h-10 w-0.5 bg-gray-300 mb-2"></div>
	// 							<div className="text-sm text-gray-500 mb-2">
	// 								True
	// 							</div>
	// 							<div
	// 								className={`w-52 p-2 rounded-lg text-center text-white ${
	// 									isPredictionPath(2)
	// 										? "ring-4 ring-yellow-500"
	// 										: ""
	// 								} ${
	// 									Number(rules[3][1]) >= 0
	// 										? "bg-blue-500"
	// 										: "bg-red-500"
	// 								}`}
	// 							>
	// 								{rules[3][0]}
	// 								<div className="text-xs mt-1 font-semibold">
	// 									Importance:{" "}
	// 									{Number(rules[3][1]).toFixed(4)}
	// 								</div>
	// 							</div>
	// 						</div>

	// 						<div className="flex flex-col items-center">
	// 							<div className="h-10 w-0.5 bg-gray-300 mb-2"></div>
	// 							<div className="text-sm text-gray-500 mb-2">
	// 								False
	// 							</div>
	// 							<div
	// 								className={`w-52 p-2 rounded-lg text-center text-white ${
	// 									Number(rules[4][1]) >= 0
	// 										? "bg-blue-500"
	// 										: "bg-red-500"
	// 								}`}
	// 							>
	// 								{rules[4][0]}
	// 								<div className="text-xs mt-1 font-semibold">
	// 									Importance:{" "}
	// 									{Number(rules[4][1]).toFixed(4)}
	// 								</div>
	// 							</div>
	// 						</div>
	// 					</div>

	// 					{/* Final prediction */}
	// 					<div className="flex flex-col items-center mt-2">
	// 						<div className="h-10 w-0.5 bg-gray-300 mb-4"></div>
	// 						<div
	// 							className={`w-32 h-32 rounded-full flex items-center justify-center text-4xl font-bold ${
	// 								selectedSample.prediction.label === 1
	// 									? "bg-blue-100 text-blue-600 border-4 border-blue-500"
	// 									: "bg-red-100 text-red-600 border-4 border-red-500"
	// 							}`}
	// 						>
	// 							{selectedSample.prediction.label}
	// 						</div>
	// 					</div>
	// 				</div>
	// 			</div>

	// 			<div className="mt-6 bg-yellow-50 p-4 rounded-lg border border-yellow-200 max-w-lg">
	// 				<h4 className="font-semibold text-lg mb-2">
	// 					Prediction Path Explanation
	// 				</h4>
	// 				<p className="text-sm">
	// 					The highlighted path shows the most influential decision
	// 					rules leading to the prediction of
	// 					<span
	// 						className={`font-bold ${
	// 							selectedSample.prediction.label === 1
	// 								? "text-blue-600"
	// 								: "text-red-600"
	// 						}`}
	// 					>
	// 						{" "}
	// 						Class {selectedSample.prediction.label}
	// 					</span>
	// 					.
	// 					{isPredictionPath(0) && Number(rules[0][1]) > 0
	// 						? ` The positive influence of "${
	// 								rules[0][0]
	// 						  }" (${Number(rules[0][1]).toFixed(
	// 								4,
	// 						  )}) indicates this condition strongly supports the prediction.`
	// 						: ` The negative influence of "${
	// 								rules[0][0]
	// 						  }" (${Number(rules[0][1]).toFixed(
	// 								4,
	// 						  )}) indicates this condition strongly opposes the prediction.`}
	// 				</p>
	// 			</div>
	// 		</div>
	// 	);
	// };
	// Replace the renderEnhancedDecisionTree function with this version
	const renderEnhancedDecisionTree = () => {
		// Sort rules by absolute value for importance
		const rules = [...selectedSample.decisionRules].sort(
			(a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1])),
		);

		// Flag for the prediction path highlighting
		const isPredictionPath = (index: number) =>
			index < Math.min(3, rules.length); // Highlight up to 3 nodes in the path

		// Check how many rules we have
		const ruleCount = rules.length;

		// Generate placeholder rules if we have fewer than 5 rules
		// This ensures our tree visualization doesn't break
		const paddedRules = [...rules];
		while (paddedRules.length < 5) {
			paddedRules.push(["No rule available", 0]);
		}

		return (
			<div className="flex flex-col items-center mt-6 relative overflow-x-auto w-full">
				<div className="w-full overflow-x-auto pb-6">
					<div className="flex flex-col items-center min-w-max">
						{/* Root node */}
						<div className="w-20 h-20 rounded-full bg-gray-100 border-2 border-gray-300 flex items-center justify-center mb-4 text-sm font-medium">
							Root
						</div>

						{/* First level */}
						{ruleCount >= 1 ? (
							<>
								<div className="h-10 w-0.5 bg-gray-300 -mt-4 mb-2"></div>
								<div
									className={`w-64 p-3 rounded-lg mb-6 text-center text-white ${
										isPredictionPath(0)
											? "ring-4 ring-yellow-500"
											: ""
									} ${
										Number(paddedRules[0][1]) >= 0
											? "bg-blue-500"
											: "bg-red-500"
									}`}
								>
									{paddedRules[0][0]}
									<div className="text-xs mt-1 font-semibold">
										Importance:{" "}
										{Number(paddedRules[0][1]).toFixed(4)}
									</div>
								</div>
							</>
						) : null}

						{/* Second level */}
						{ruleCount >= 2 ? (
							<div className="flex justify-center w-full mb-6">
								<div className="flex flex-col items-center mx-8">
									<div className="h-10 w-0.5 bg-gray-300 mb-2"></div>
									<div className="text-sm text-gray-500 mb-2">
										True
									</div>
									<div
										className={`w-60 p-3 rounded-lg text-center text-white ${
											isPredictionPath(1)
												? "ring-4 ring-yellow-500"
												: ""
										} ${
											Number(paddedRules[1][1]) >= 0
												? "bg-blue-500"
												: "bg-red-500"
										}`}
									>
										{paddedRules[1][0]}
										<div className="text-xs mt-1 font-semibold">
											Importance:{" "}
											{Number(paddedRules[1][1]).toFixed(
												4,
											)}
										</div>
									</div>
								</div>

								{ruleCount >= 3 ? (
									<div className="flex flex-col items-center mx-8">
										<div className="h-10 w-0.5 bg-gray-300 mb-2"></div>
										<div className="text-sm text-gray-500 mb-2">
											False
										</div>
										<div
											className={`w-60 p-3 rounded-lg text-center text-white ${
												isPredictionPath(2)
													? "ring-4 ring-yellow-500"
													: ""
											} ${
												Number(paddedRules[2][1]) >= 0
													? "bg-blue-500"
													: "bg-red-500"
											}`}
										>
											{paddedRules[2][0]}
											<div className="text-xs mt-1 font-semibold">
												Importance:{" "}
												{Number(
													paddedRules[2][1],
												).toFixed(4)}
											</div>
										</div>
									</div>
								) : (
									<div className="flex flex-col items-center mx-8">
										<div className="h-10 w-0.5 bg-gray-300 mb-2"></div>
										<div className="text-sm text-gray-500 mb-2">
											False
										</div>
										<div className="w-60 p-3 rounded-lg text-center text-white bg-gray-400">
											No additional rules
										</div>
									</div>
								)}
							</div>
						) : null}

						{/* Third level - Only show if we have enough rules */}
						{ruleCount >= 4 ? (
							<div className="flex justify-center w-full space-x-4 mb-6">
								<div className="flex flex-col items-center">
									<div className="h-10 w-0.5 bg-gray-300 mb-2"></div>
									<div className="text-sm text-gray-500 mb-2">
										True
									</div>
									<div
										className={`w-52 p-2 rounded-lg text-center text-white ${
											isPredictionPath(3)
												? "ring-4 ring-yellow-500"
												: ""
										} ${
											Number(paddedRules[3][1]) >= 0
												? "bg-blue-500"
												: "bg-red-500"
										}`}
									>
										{paddedRules[3][0]}
										<div className="text-xs mt-1 font-semibold">
											Importance:{" "}
											{Number(paddedRules[3][1]).toFixed(
												4,
											)}
										</div>
									</div>
								</div>

								{ruleCount >= 5 ? (
									<div className="flex flex-col items-center">
										<div className="h-10 w-0.5 bg-gray-300 mb-2"></div>
										<div className="text-sm text-gray-500 mb-2">
											False
										</div>
										<div
											className={`w-52 p-2 rounded-lg text-center text-white ${
												Number(paddedRules[4][1]) >= 0
													? "bg-blue-500"
													: "bg-red-500"
											}`}
										>
											{paddedRules[4][0]}
											<div className="text-xs mt-1 font-semibold">
												Importance:{" "}
												{Number(
													paddedRules[4][1],
												).toFixed(4)}
											</div>
										</div>
									</div>
								) : (
									<div className="flex flex-col items-center">
										<div className="h-10 w-0.5 bg-gray-300 mb-2"></div>
										<div className="text-sm text-gray-500 mb-2">
											False
										</div>
										<div className="w-52 p-2 rounded-lg text-center text-white bg-gray-400">
											No additional rules
										</div>
									</div>
								)}
							</div>
						) : null}

						{/* Final prediction */}
						<div className="flex flex-col items-center mt-2">
							<div className="h-10 w-0.5 bg-gray-300 mb-4"></div>
							<div
								className={`w-32 h-32 rounded-full flex items-center justify-center text-4xl font-bold ${
									selectedSample.prediction.label === 1
										? "bg-blue-100 text-blue-600 border-4 border-blue-500"
										: "bg-red-100 text-red-600 border-4 border-red-500"
								}`}
							>
								{selectedSample.prediction.label}
							</div>
						</div>
					</div>
				</div>

				<div className="mt-6 bg-yellow-50 p-4 rounded-lg border border-yellow-200 max-w-lg">
					<h4 className="font-semibold text-lg mb-2">
						Prediction Path Explanation
					</h4>
					<p className="text-sm">
						The highlighted path shows the most influential decision
						rules leading to the prediction of
						<span
							className={`font-bold ${
								selectedSample.prediction.label === 1
									? "text-blue-600"
									: "text-red-600"
							}`}
						>
							{" "}
							Class {selectedSample.prediction.label}
						</span>
						.
						{ruleCount > 0 &&
						isPredictionPath(0) &&
						Number(paddedRules[0][1]) > 0
							? ` The positive influence of "${
									paddedRules[0][0]
							  }" (${Number(paddedRules[0][1]).toFixed(
									4,
							  )}) indicates this condition strongly supports the prediction.`
							: ruleCount > 0
							? ` The negative influence of "${
									paddedRules[0][0]
							  }" (${Number(paddedRules[0][1]).toFixed(
									4,
							  )}) indicates this condition strongly opposes the prediction.`
							: ""}
					</p>
				</div>
			</div>
		);
	};

	return (
		<div className="p-6 max-w-6xl mx-auto space-y-6 bg-gray-50">
			<h1 className="text-3xl font-bold text-center mb-6">
				Explainable AI Dashboard
			</h1>

			{/* Controls */}
			<div className="bg-white shadow-md rounded-lg p-6">
				<div className="grid grid-cols-1 md:grid-cols-3 gap-4">
					<div>
						<label className="block text-sm font-medium text-gray-700 mb-2">
							Dataset
						</label>
						<select
							className="w-full p-2 border border-gray-300 rounded-md bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
							value={selectedDataset}
							onChange={(e) => setSelectedDataset(e.target.value)}
						>
							{datasets.map((dataset) => (
								<option key={dataset} value={dataset}>
									{dataset}
								</option>
							))}
						</select>
					</div>

					<div>
						<label className="block text-sm font-medium text-gray-700 mb-2">
							Model
						</label>
						<select
							className="w-full p-2 border border-gray-300 rounded-md bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
							value={selectedModel}
							onChange={(e) => setSelectedModel(e.target.value)}
						>
							{models.map((model) => (
								<option key={model} value={model}>
									{model}
								</option>
							))}
						</select>
					</div>

					<div>
						<label className="block text-sm font-medium text-gray-700 mb-2">
							Sample
						</label>
						<select
							className="w-full p-2 border border-gray-300 rounded-md bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
							value={selectedSample.id}
							onChange={(e) => {
								const sample =
									samples.find(
										(s) => s.id === e.target.value,
									) || samples[0];
								setSelectedSample(sample);
							}}
						>
							{samples.map((sample) => (
								<option key={sample.id} value={sample.id}>
									{sample.name}
								</option>
							))}
						</select>
					</div>
				</div>
			</div>

			{/* Model Metrics Box (New) */}
			<div className="bg-white shadow-md rounded-lg p-6">
				<h2 className="text-xl font-semibold mb-4">
					Model Performance Metrics
				</h2>
				<div className="grid grid-cols-2 md:grid-cols-2 gap-6">
					<div className="bg-blue-50 rounded-lg p-4 text-center border border-blue-200">
						<div className="text-sm text-gray-600 mb-1">
							Black Box Accuracy
						</div>
						<div className="text-3xl font-bold text-blue-700">
							{(selectedSample.accuracy * 100).toFixed(1)}%
						</div>
					</div>
					<div className="bg-green-50 rounded-lg p-4 text-center border border-green-200">
						<div className="text-sm text-gray-600 mb-1">
							Black Box F1 Score
						</div>
						<div className="text-3xl font-bold text-green-700">
							{(selectedSample.f1Score * 100).toFixed(1)}%
						</div>
					</div>
				</div>
			</div>

			{/* Sample Data and Prediction */}
			<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
				{/* Sample Data Card */}
				<div className="bg-white shadow-md rounded-lg p-6">
					<h2 className="text-xl font-semibold mb-4">
						Sample Input Data
					</h2>
					<div className="overflow-x-auto">
						<table className="w-full border-collapse">
							<thead>
								<tr className="bg-gray-100">
									<th className="border p-2 text-left">
										Feature
									</th>
									<th className="border p-2 text-left">
										Value
									</th>
								</tr>
							</thead>
							<tbody>
								{Object.entries(selectedSample.data).map(
									([key, value]) => (
										<tr
											key={key}
											className="hover:bg-gray-50"
										>
											<td className="border p-2 font-medium">
												{key}
											</td>
											<td className="border p-2">
												{value}
											</td>
										</tr>
									),
								)}
							</tbody>
						</table>
					</div>
				</div>

				{/* Prediction Card */}
				<div className="bg-white shadow-md rounded-lg p-6">
					<h2 className="text-xl font-semibold mb-4">
						Model Prediction
					</h2>
					<div className="flex items-center justify-center mb-6">
						<div
							className={`text-5xl font-bold rounded-full h-24 w-24 flex items-center justify-center ${
								selectedSample.prediction.label === 1
									? "bg-blue-100 text-blue-600"
									: "bg-red-100 text-red-600"
							}`}
						>
							{selectedSample.prediction.label}
						</div>
					</div>

					<h3 className="text-lg font-medium mb-2">
						Prediction Probabilities
					</h3>
					<div className="grid grid-cols-2 gap-4">
						<div className="bg-gray-100 p-4 rounded-md">
							<div className="text-sm text-gray-500">Class 0</div>
							<div className="text-lg font-semibold">
								{(
									selectedSample.prediction.probabilities[0] *
									100
								).toFixed(2)}
								%
							</div>
							<div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
								<div
									className="bg-red-600 h-2.5 rounded-full"
									style={{
										width: `${
											selectedSample.prediction
												.probabilities[0] * 100
										}%`,
									}}
								></div>
							</div>
						</div>

						<div className="bg-gray-100 p-4 rounded-md">
							<div className="text-sm text-gray-500">Class 1</div>
							<div className="text-lg font-semibold">
								{(
									selectedSample.prediction.probabilities[1] *
									100
								).toFixed(2)}
								%
							</div>
							<div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
								<div
									className="bg-blue-600 h-2.5 rounded-full"
									style={{
										width: `${
											selectedSample.prediction
												.probabilities[1] * 100
										}%`,
									}}
								></div>
							</div>
						</div>
					</div>
				</div>
			</div>

			{/* Feature Importance */}
			<div className="bg-white shadow-md rounded-lg p-6 ">
				<h2 className="text-3xl font-semibold mb-4">
					Feature Importance
				</h2>

				<div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6">
					<div className="bg-amber-50 rounded-lg p-4 text-center border border-amber-200">
						<div className="text-sm text-gray-600 mb-1">
							Explanation Fidelity
						</div>
						<div className="text-3xl font-bold text-amber-700">
							{(selectedSample.fidelity_lr * 100).toFixed(1)}%
						</div>
						<div className="text-xs text-gray-500 mt-2">
							How well explanations match model behavior
						</div>
					</div>
					<div className="bg-teal-50 rounded-lg p-4 text-center border border-teal-200">
						<div className="text-sm text-gray-600 mb-1">
							Neighbourhood Fidelity
						</div>
						<div className="text-3xl font-bold text-teal-700">
							{(selectedSample.localFidelity_lr * 100).toFixed(1)}
							%
						</div>
						<div className="text-xs text-gray-500 mt-2">
							Accuracy of the surrogate model on the Neighbourhood
							dataset
						</div>
					</div>
				</div>
				<ResponsiveContainer width="100%" height={600}>
					<BarChart
						data={getFeatureImportanceData()}
						layout="vertical"
					>
						<CartesianGrid strokeDasharray="3 3" />
						<XAxis type="number" />
						<YAxis dataKey="feature" type="category" width={120} />
						<Tooltip />
						<Bar
							dataKey="importance"
							fill="#000000FF"
							label={{
								position: "right",
								formatter: (value: number) => value.toFixed(4),
							}}
						/>
					</BarChart>
				</ResponsiveContainer>
			</div>

			{/* Enhanced Decision Rules */}
			<div className="bg-white shadow-md rounded-lg p-6">
				<h2 className="text-3xl font-semibold mb-4">Decision Rules</h2>

				<div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6">
					<div className="bg-amber-50 rounded-lg p-4 text-center border border-amber-200">
						<div className="text-sm text-gray-600 mb-1">
							Explanation Fidelity
						</div>
						<div className="text-3xl font-bold text-amber-700">
							{(selectedSample.fidelity_dt * 100).toFixed(1)}%
						</div>
						<div className="text-xs text-gray-500 mt-2">
							How well explanations match model behavior
						</div>
					</div>
					<div className="bg-teal-50 rounded-lg p-4 text-center border border-teal-200">
						<div className="text-sm text-gray-600 mb-1">
							Neighbourhood Fidelity
						</div>
						<div className="text-3xl font-bold text-teal-700">
							{(selectedSample.localFidelity_dt * 100).toFixed(1)}
							%
						</div>
						<div className="text-xs text-gray-500 mt-2">
							Accuracy of the surrogate model on the Neighbourhood
							dataset
						</div>
					</div>
				</div>
				<div className="grid grid-cols-1 gap-6">
					<div>
						<h3 className="text-lg font-medium mb-3">
							Decision Tree Visualization
						</h3>
						{renderEnhancedDecisionTree()}
					</div>
					<div>
						<h3 className="text-lg font-medium mb-3 mt-6">
							All Decision Rules
						</h3>
						<div className="overflow-y-auto max-h-96">
							<table className="w-full border-collapse">
								<thead>
									<tr className="bg-gray-100">
										<th className="border p-2 text-left">
											Rule
										</th>
										<th className="border p-2 text-left">
											Value
										</th>
									</tr>
								</thead>
								<tbody>
									{selectedSample.decisionRules.map(
										(rule, idx) => (
											<tr
												key={idx}
												className={`hover:bg-gray-50 ${
													idx < 3
														? "bg-yellow-50"
														: ""
												}`}
											>
												<td className="border p-2">
													{rule[0]}
												</td>
												<td
													className={`border p-2 font-medium ${
														Number(rule[1]) >= 0
															? "text-blue-600"
															: "text-red-600"
													}`}
												>
													{Number(rule[1]).toFixed(4)}
												</td>
											</tr>
										),
									)}
								</tbody>
							</table>
						</div>
					</div>
				</div>
			</div>

			{/* Similar Samples */}
			<div className="bg-white shadow-md rounded-lg p-6">
				<h2 className="text-xl font-semibold mb-4">
					Exemplar-Based Explanation
				</h2>
				<p className="text-gray-700 mb-4">
					Similar samples from the training dataset that influence
					this prediction:
				</p>

				<div className="grid grid-cols-1 md:grid-cols-3 gap-6">
					{selectedSample.similarSamples.map((sample, index) => (
						<div
							key={index}
							className="border rounded-lg p-4 bg-gray-50"
						>
							<div className="mb-2">
								<span className="font-medium">
									Similar Sample {index + 1}
								</span>
							</div>
							<div className="text-xs text-gray-500 mb-1">
								ID: {sample.id}
							</div>
							<div className="overflow-y-auto max-h-48">
								<table className="w-full border-collapse text-sm">
									<thead>
										<tr className="bg-gray-100">
											<th className="border p-1 text-left">
												Feature
											</th>
											<th className="border p-1 text-left">
												Value
											</th>
										</tr>
									</thead>
									<tbody>
										{Object.entries(sample.data).map(
											([key, value]) => (
												<tr
													key={key}
													className="hover:bg-white"
												>
													<td className="border p-1">
														{key}
													</td>
													<td className="border p-1">
														{value}
													</td>
												</tr>
											),
										)}
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
