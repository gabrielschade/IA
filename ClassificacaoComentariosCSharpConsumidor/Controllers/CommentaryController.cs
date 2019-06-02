using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ClassificacaoComentariosCSharpConsumidor.Types;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;

namespace ClassificacaoComentariosCSharpConsumidor.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class CommentaryController : ControllerBase
    {
        [HttpGet("{text}")]
        public ActionResult<PredictionResult> Get(string text)
        {
            string diretorioModelo = "Model\\model.zip";

            MLContext mlContext = new MLContext();
            ITransformer model;

            using (var stream = new FileStream(diretorioModelo, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                model = mlContext.Model.Load(stream, out DataViewSchema modelSchema);
            }

            var predictiveModel = 
                mlContext.Model.CreatePredictionEngine<Commentary, PredictionResult>(model);

            return predictiveModel.Predict(new Commentary()
            {
                Text = text
            });
        }

    }
}
