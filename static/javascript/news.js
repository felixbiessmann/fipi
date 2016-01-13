var news;
var topics;

d3.json("/api/news",function(error, json) {
    if (error) return console.warn(error);
	news = json

console.log(news)

d3.json("/api/newstopics",function(error, json) {
    if (error) return console.warn(error);
	topics = json
	var newsContainer  = d3.select("#news")
	newsContainer.selectAll("#posts").remove()

	newsContainer.selectAll("#posts")
	.data(topics)
	.enter()
	.append("div")
	.attr("class","posts")
	.html(function(d, i){
		var clusterid = "<h1 class=\"content-subhead\">Topic "+ i +"</h1>"
		var head =  "<section class=\"post\"><header class=\"post-header\">"
					+ "<h3 class=\"post-title\">"+ d.description.split(" ").slice(0,4).join(" ") + "</h3>"
                    + "</header><post-description>"
		var content = d.members
						.map(function(i){return news[i];})
						.sort(function(a,b){
							var lefta = a[1].leftright[0].prediction-a[1].leftright[1].prediction
							var leftb = b[1].leftright[0].prediction-b[1].leftright[1].prediction
							return lefta - leftb
						})
						.map(
			function(article) {
			var leftrightPrediction = article[1].leftright.map(function(p){return " "+p.label + ": " + Math.round(p.prediction * 100) / 100;})
			console.log(leftrightPrediction)
			var itemhead = "<p><a href=\"" + article[1].url + "\">" + article[0].slice(0,40)
				+ "... | " + leftrightPrediction +  "</a></p>"
			return itemhead
			}
		).join("") + "</post-description>"
	return clusterid + head + content
	})
})

})


