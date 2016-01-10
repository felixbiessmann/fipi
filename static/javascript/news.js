var news;
var topics;

d3.json("/api/news",function(error, json) {
    if (error) return console.warn(error);
    console.log(news)
	news = json
})


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
//                         <img class="post-avatar" alt="Tilo Mitra&#x27;s avatar" height="48" width="48" src="img/common/tilo-avatar.png">
					+ "<h3 class=\"post-title\">"+ d.description.split(" ").slice(0,4).join(" ") + "</h3>"

//                         <p class="post-meta">
//                             By <a href="#" class="post-author">Tilo Mitra</a> under <a class="post-category post-category-design" href="#">CSS</a> <a class="post-category post-category-pure" href="#">Pure</a>
//                         </p>
                    + "</header><post-description>"

		var content = d.members.map(
			function(i) {
			//console.log(i)
			var itemhead = "<p><a href=\"" + news[i][1]["url"] + "\">" + news[i][0] + "</a></p>"

			return itemhead
			}
		).join("") + "</post-description>"
	return clusterid + head + content
	})
})



