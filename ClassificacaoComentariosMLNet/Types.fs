module Types

open Microsoft.ML.Data

[<CLIMutable>]
type DataCommentary =    {
    [<LoadColumn(0)>]
    Text : string

    [<LoadColumn(1)>]
    Label : bool
}   

[<CLIMutable>]
type InputCommentary = {
    Text:string
}

[<CLIMutable>]
type Prediction = {
    [<ColumnName("PredictedLabel")>]
    Prediction : bool

    Score : float32
}
