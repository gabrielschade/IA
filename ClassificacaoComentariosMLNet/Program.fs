open System
open Types
open Microsoft.ML
open Microsoft.ML.Data
open System.IO
open XPlot.GoogleCharts
open System.Diagnostics

let preprocessing (mlContext:MLContext) = 
    let textLoader = mlContext.Data.CreateTextLoader<DataCommentary>(',', true)
    let rawData = textLoader.Load 
                    [|
                      "Datasets/amazon_cells_labelled.csv"
                      "Datasets/imdb_labelled.csv"
                      "Datasets/yelp_labelled.csv"
                    |]
    let data = mlContext.Data.TrainTestSplit(rawData, 0.2)

    rawData, data

let train (mlContext:MLContext) 
          (data:IDataView)
          (trainer:IEstimator<'u>)=

    let mlFeaturedText = 
        mlContext.Transforms.Text.FeaturizeText("Features", "Text")

    let pipeline = mlFeaturedText.Append trainer
    pipeline.Fit data

let evaluate (mlContext:MLContext) 
             (data:IDataView) 
             (model:TransformerChain<'v>) =

    let predictions = model.Transform data
    let metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(
                    predictions, 
                    "Label", 
                    "Score"
                )
    [|
        "Accuracy", metrics.Accuracy
        "Positive Precision", metrics.PositivePrecision
        "Negative Precision", metrics.NegativePrecision
        "Positive Recall", metrics.PositiveRecall
        "Negative Recall", metrics.NegativeRecall
        "F1 Score", metrics.F1Score
    |]

[<EntryPoint>]
let main argv =
    let mlContext = new MLContext()
    let (raw, datasets) = preprocessing mlContext
    let trainWith = train mlContext datasets.TrainSet
    let evaluateWith = evaluate mlContext datasets.TestSet

    let trainAndEvaluate (trainer:IEstimator<'u>)=
        trainWith trainer
        |> evaluateWith 

    
    let trainerSgd = mlContext.BinaryClassification.Trainers.SgdNonCalibrated(
                        "Label", 
                        "Features")

    let trainerPerceptron = mlContext.BinaryClassification.Trainers.AveragedPerceptron(
                                "Label", 
                                "Features")

    let trainerSvm = mlContext.BinaryClassification.Trainers.LinearSvm(
                        "Label", 
                        "Features")

    let trainerSdca = mlContext.BinaryClassification.Trainers.SdcaNonCalibrated(
                        "Label", 
                        "Features")

    let trainersLabels = [
        "SgdNonCalibrated"
        "AveragedPerceptron"
        "LinearSvm"
        "SdcaNonCalibrated"
    ]

    let metrics = [
        trainAndEvaluate trainerSgd
        trainAndEvaluate trainerPerceptron
        trainAndEvaluate trainerSvm
        trainAndEvaluate trainerSdca
    ]
    
    let options = Options ( title = "Metrics Comparison", 
                            hAxis = Axis(
                                title = "Algorithm",
                                titleTextStyle = TextStyle(color = "blue")
                            ))

    let chart = 
        metrics
        |> Chart.Column
        |> Chart.WithOptions options
        |> Chart.WithLabels trainersLabels

    let html = chart.GetHtml()
    File.AppendAllLines ("metrics.html",[html])

    Process.Start (@"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", 
                    "file:\\" + Directory.GetCurrentDirectory() + "\\metrics.html")
    |> ignore  

    
    let model = trainWith trainerSdca
    mlContext.Model.Save(model, raw.Schema,"model.zip")
    

    Console.ReadKey() |> ignore

    //Console.WriteLine data
    0 // return an integer exit code
