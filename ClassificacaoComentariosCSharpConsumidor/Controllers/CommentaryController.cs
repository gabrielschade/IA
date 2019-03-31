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
            string diretorioBase = @"C:\Users\re035148\Documents\GitHub\IA\ClassificacaoComentariosCSharpConsumidor\";
            string diretorioModelo = string.Concat(diretorioBase, "\\Model\\model.zip");

            MLContext mlContext = new MLContext();
            using (var stream = new FileStream(diretorioModelo, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                var model = mlContext.Model.Load(stream);
                var predictiveModel = model.CreatePredictionEngine<Commentary, PredictionResult>(mlContext);
                return predictiveModel.Predict(new Commentary()
                {
                    Text = text
                });
            }
        }

    }
}
