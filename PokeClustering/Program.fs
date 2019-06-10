open System
open System.IO
open System.Diagnostics
open Types
open Microsoft.ML
open Microsoft.ML.Data
open FSharp.Data
open XPlot.Plotly

[<Literal>]
let url = "Dataset/pokemon.csv"
type PokemonCSV = CsvProvider<url>

let typeToSingle pkmnType = 
    match pkmnType with
    |"Bug" -> 0
    |"Dark"-> 1
    |"Dragon"-> 2
    |"Electric"-> 3
    |"Fairy"-> 4
    |"Fighting"-> 5
    |"Fire"-> 6
    |"Flying"-> 7
    |"Ghost"-> 8
    |"Grass"-> 9
    |"Ground"-> 10
    |"Ice"-> 11
    |"Normal"-> 12
    |"Poison"-> 13
    |"Psychic"-> 14
    |"Rock"-> 15
    |"Stell"-> 16
    |"Water"-> 17
    |_ -> -1
    |> (single)

[<EntryPoint>]
let main argv =
    let allPokemon = PokemonCSV.Load url
    let listOfPokemon = allPokemon.Rows
                        |> Seq.map (fun pokemon -> {
                            Number = (single) pokemon.Number
                            Name = pokemon.Name 
                            Type1 = pokemon.``Type 1``
                            ConvertedType1 = (typeToSingle pokemon.``Type 1``)
                            Type2 = pokemon.``Type 2``
                            ConvertedType2 = (typeToSingle pokemon.``Type 2``)
                            HP = (single) pokemon.HP
                            Attack = (single) pokemon.Attack
                            SpAttack = (single) pokemon.``Sp. Atk``
                            Defense = (single) pokemon.Defense
                            SpDefense = (single) pokemon.``Sp. Def``
                            Speed = (single) pokemon.Speed
                            Total = (single) pokemon.Total
                        })

    let mlContext = new MLContext();
    let data = listOfPokemon
               |> mlContext.Data.LoadFromEnumerable
    
    let pipeline = EstimatorChain().Append(
                           mlContext.Transforms.Concatenate( "Features", 
                               "ConvertedType1","ConvertedType2",
                               "HP","Attack","SpAttack", 
                               "Defense", "SpDefense", "Speed","Total" ))    

    let options = Trainers.KMeansTrainer.Options()
    options.NumberOfClusters <- 3
    options.FeatureColumnName <- "Features"
    
    let trainer = mlContext.Clustering.Trainers.KMeans options
    let pipelineTraining = pipeline.Append trainer
    let model = pipelineTraining.Fit data

    //let model = mlContext.Model.Load("model.zip", ref(data.Schema))
    let predictiveModel = 
        mlContext.Model.CreatePredictionEngine<Pokemon, ClusterPrediction>(model)
    
    let clusterizedList = 
        listOfPokemon
        |> Seq.map(fun pokemon -> pokemon,(predictiveModel.Predict pokemon))
        
    printfn "Number; Name; Total; Type1; Type2; Cluster"
        
    clusterizedList
    |> Seq.iter(fun (pkmn, result) -> printfn "%i;%s;%i;%s;%s;%i" 
                                                ((int)pkmn.Number) 
                                                pkmn.Name 
                                                ((int)pkmn.Total)
                                                pkmn.Type1
                                                pkmn.Type2
                                                result.PredictedClusterId
                )

    let chartData = [
        for cluster in 0..options.NumberOfClusters-1 do
            let pkmn = clusterizedList 
                        |> Seq.filter( fun (pkmn, result) -> result.PredictedClusterId = (uint32) cluster+1u)
            yield Scatter3d(
                    x = (pkmn |> Seq.map( fun (pkmn, result) -> pkmn.ConvertedType1)),
                    y = (pkmn |> Seq.map( fun (pkmn, result) -> pkmn.ConvertedType2)),
                    z = (pkmn |> Seq.map( fun (pkmn, result) -> pkmn.Total)),
                    text = (pkmn |> Seq.map( fun (pkmn, result) -> pkmn.Name)),
                    mode = "markers",
                    marker =
                        Marker(
                            size = 12.,
                            opacity = 0.8
                        )
            )
    ]

    
    let chartOptions = Options ( title = "Pokémon Cluster")
    
    let chart = 
        chartData
        |> Chart.Plot
        |> Chart.WithOptions chartOptions
        |> Chart.WithHeight 600
        |> Chart.WithWidth 800
        |> Chart.WithLabels ["Pokémon mais fracos"; "Pokémon";"Pokémon muito poderosos"]

    //let html = chart.GetHtml()
    //File.AppendAllLines ("metrics.html",[html])
    
    //Process.Start (@"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", 
    //                "file:\\" + Directory.GetCurrentDirectory() + "\\metrics.html")
    //                |> ignore

    //mlContext.Model.Save(model, data.Schema,"model.zip")

    Console.ReadKey() |> ignore
    0
