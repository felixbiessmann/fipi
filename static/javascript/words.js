function queryPosts(queryTextElement){
  wordlist = queryTextElement.value
	d3.json("/api/postquery")
		.header("Content-type", "application/x-www-form-urlencoded")
		.post(
      "partyRadio="
      + d3.select('input[name="partyRadio"]:checked').property("value")
      + "&wordlist=" + wordlist, function(error, postTexts) {
      var postContainers = d3.select("#news")
      .selectAll("div.posts").remove()

      var words = wordlist.replace(","," ").split(" ")

      postContainers
      .data(postTexts)
      .enter()
      .append("div")
      .attr("class","posts")
      .html(function(d, i){
        content = "<div style=\"background:#eef7f3;border:1px solid #eef7f3;padding:0.5em;margin:0.5em;\">"
            + "<p style=\"padding:0.1em;margin:0.1em;\">"
            + words.map(
              function(w){
              return  "... " +
                d.text.substr(Math.max(0,d.text.indexOf(w) - 200),400)
                + " ..."
              })
            + "</p>"
            + "</div>"
            return content
        })


  })
  }
