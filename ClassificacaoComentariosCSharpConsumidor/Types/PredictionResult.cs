using Microsoft.ML.Data;

namespace ClassificacaoComentariosCSharpConsumidor.Types
{
    public class PredictionResult
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Score { get; set; }
    }
}
