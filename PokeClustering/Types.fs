module Types

open Microsoft.ML.Data

type Pokemon = {
    Number : single
    Name : string
    Type1: string
    Type2 : string
    ConvertedType1:single
    ConvertedType2:single
    Total: single
    HP : single
    Attack : single
    Defense : single
    SpAttack : single
    SpDefense : single
    Speed : single
}   

[<CLIMutable>]
type ClusterPrediction = {
    [<ColumnName("PredictedLabel")>] 
    PredictedClusterId : uint32

    [<ColumnName("Score")>] 
    Distances : single array
}