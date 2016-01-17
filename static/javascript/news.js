var news;
var topics;

var icons = {"spiegel":"http://img.informer.com/icons/png/32/150/150726.png"}

d3.json("/api/news",function(error, json) {
    if (error) return console.warn(error);
	news = json

function getMaxPredictions(p,nMax) {
	var sorted = p.sort(function(a,b){return b.prediction-a.prediction});
	return sorted.slice(0,nMax).map(function(i){return i.label + " (" + Math.round(i.prediction*100) + "%)"});
}

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
			var itemhead = "<div style=\"background:#eef7f3;border:1px solid #eef7f3;padding:0.5em;margin:0.5em;\">"
				+ "<p style=\"padding:0.1em;margin:0.1em;\"><a href=\"" + article[1].url + "\">" + article[0].slice(0,80)
				+ " ...  </a></p>"
				+ "<div>"
				+ "<div class=\"post-category post-category-design\">" + article[1].source + "</div>"
				+ "<div class=\"post-category post-category-yui\"\">" + getMaxPredictions(article[1].leftright,1) + "</div>"
				+ "<div class=\"post-category post-category-pure\">"+ getMaxPredictions(article[1].domain,1) + "</div>"
				+ "<div class=\"post-category post-category-js\">" + getMaxPredictions(article[1].manifestocode,1) +  "</div>"
				+ "</div></div>"
			return itemhead
			}
		).join("") + "</post-description>"
	return clusterid + head + content
	})
})

})


