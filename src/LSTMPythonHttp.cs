using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BidAskLast.Server.Common.DTOs;
using BidAskLast.Server.Tiers.Application.WebApi.Controllers;
using BidAskLast.Server.Tiers.Core.MarketData;
using BidAskLast.Server.Tiers.Domain.TA;
using Newtonsoft.Json;
using System.Globalization;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;
using CsvHelper;

namespace BidAskLast.Server.MachineLearning.LSTM
{
    public class LSTMPythonHttp
    {

        public class SequenceRequest
        {
            public List<List<float>> sequence { get; set; } = [];
        }

        public class PredictionResponse
        {
            public float prediction { get; set; }
        }

        public class LstmApiClient
        {
            private readonly HttpClient _client;
            private readonly string _apiUrl;

            public LstmApiClient(string apiUrl)
            {
                _client = new HttpClient();
                _apiUrl = apiUrl;
            }

            public async Task<float> PredictAsync(List<List<float>> sequence)
            {
                var request = new SequenceRequest { sequence = sequence };
                var response = await _client.PostAsJsonAsync(_apiUrl, request);
                response.EnsureSuccessStatusCode();
                var result = await response.Content.ReadFromJsonAsync<PredictionResponse>();
                if (result == null)
                    throw new InvalidOperationException("PredictionResponse was null.");
                return result.prediction;
            }
        }

        public static List<List<List<float>>> ReadSequencesWithContinuity(
        string csvPath, int seqLength, int numFeatures,
        int tradingDayCol = 0, int msCol = 1, int featureStartCol = 5, int featureEndCol = 48, int targetCol = -1)
        {
            var allRows = new List<(string TradingDay, long Ms, List<float> Features)>();
            using (var reader = new StreamReader(csvPath))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                csv.Read();
                csv.ReadHeader();
                while (csv.Read())
                {
                    string tradingDay = csv.GetField(tradingDayCol) ?? string.Empty;
                    long ms = (long)csv.GetField<double>(msCol);
                    var features = new List<float>();
                    for (int colIdx = featureStartCol; colIdx <= featureEndCol; colIdx++)
                        features.Add(csv.GetField<float>(colIdx));
                    allRows.Add((tradingDay, ms, features));
                }
            }

            var sequences = new List<List<List<float>>>();
            int i = 0;
            while (i <= allRows.Count - seqLength)
            {
                bool isConsecutive = true;
                string currentDay = allRows[i].TradingDay;
                long currentMs = allRows[i].Ms;
                for (int j = 1; j < seqLength; j++)
                {
                    if (allRows[i + j].TradingDay != currentDay ||
                        allRows[i + j].Ms != currentMs + j * 60000)
                    {
                        isConsecutive = false;
                        break;
                    }
                }
                if (isConsecutive)
                {
                    var sequence = new List<List<float>>();
                    for (int k = 0; k < seqLength; k++)
                        sequence.Add(allRows[i + k].Features);
                    sequences.Add(sequence);
                    i += 1; // sliding window
                }
                else
                {
                    i += 1; // skip to next possible start
                }
            }
            return sequences;
        }

        public async Task RunAsync()
        {
            string csvPath = "Resources/testing_data_spy_20250625_20250710.csv";
            string apiUrl = "http://localhost:8300/predict";
            int seqLength = 20;
            int numFeatures = 44;

            var sequences = ReadSequencesWithContinuity(csvPath, seqLength, numFeatures);
            var client = new LstmApiClient(apiUrl);

            foreach (var sequence in sequences)
            {
                float prediction = await client.PredictAsync(sequence);
                Console.WriteLine($"API Prediction: {prediction}");
            }
        }
    }
}