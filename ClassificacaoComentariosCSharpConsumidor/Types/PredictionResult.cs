using Microsoft.ML.Data;

namespace ClassificacaoComentariosCSharpConsumidor.Types
{
    public class PredictionResult
    {
        [ColumnName("PredictedLabel")]
        public bool Predicition { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
