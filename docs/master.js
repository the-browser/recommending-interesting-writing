$(document).ready(function () {

  $('a').click(function () {
    $(this).attr('target', '_blank');
  });
  $("button").click(function () {
    var current_selected_tab = $("section.is-active");
    var current_data_from_id = current_selected_tab.attr('id').slice(10)
    current_selected_tab.find(".rank_results").remove();
    current_selected_tab.find(".spinner").show();
    var slider_values = [];
    var slider_indexes = [];
    current_selected_tab.find(".mdl-slider").each(function (index) {
      slider_values[index] = parseFloat($(this).val());
      slider_indexes[index] = parseInt($(this).attr("id").split('-')[1].slice(1));
    });
    var a = slider_values[0];
    var b = slider_values[1];
    var c = slider_values[2];
    var d = slider_values[3];
    const Url = "https://jp0pvkfc6f.execute-api.us-east-2.amazonaws.com/default/rank_news_demo";
    var Data = {
      dataset: current_data_from_id,
      a: a,
      b: b,
      c: c,
      d: d,
      idx1: slider_indexes[0],
      idx2: slider_indexes[1],
      idx3: slider_indexes[2],
      idx4: slider_indexes[3],
    };
    console.log(Data);
    var payLoad = {
      method: "POST",
      mode: "cors",
      body: JSON.stringify(Data)
    };

    (async () => {
      let ranked_articles = await fetch(Url, payLoad)
        .then(response => response.json())
        .then(data => {
          return data
        });
      var grand_html = ""
      ranked_articles.forEach(function (article) {
        grand_html += "<tr>";
        grand_html += '<td class="logit">';
        grand_html += article['logit'].toString();
        grand_html += '</td>'
        grand_html += '<td class="title mdl-data-table__cell--non-numeric"><a class="article_link" href="';
        grand_html += article['link'];
        grand_html += '">';
        grand_html += article['title'];
        grand_html += '<br></br>';
        grand_html += '<p class="publication">'
        grand_html += article['publication']
        grand_html += '</p>'
        grand_html += '<p class="top_words">';
        article['top_words'].forEach(function (word) {
          if (word.length > 2 && isNaN(word)) {
            grand_html += word;
            grand_html += ",";
          };
        });
        grand_html = grand_html.slice(0, -1)
        grand_html += '</p>';
        grand_html += '<p class="least_words">';
        article['least_words'].forEach(function (word) {
          if (word.length > 2 && isNaN(word)) {
            grand_html += word;
            grand_html += ",";
          };
        });
        grand_html = grand_html.slice(0, -1)
        grand_html += '</p>';
        grand_html += '</td>';
        grand_html += '</tr>'
      });
      grand_html += '</tbody>'
      grand_html += '</table>'
      prepend += `<table class="mdl-data-table mdl-js-data-table rank_results">
      <thead>
      <tr>
      <th class="mdl-data-table__cell--non-numeric">Score</th>
      <th class="mdl-data-table__cell--non-numeric">Title</th>
      </tr>
      </thead>`
      var final_html_str = prepend + grand_html
      current_selected_tab.find(".spinner").hide();
      current_selected_tab.find(".table-wrapper").append(final_html_str);
    })()
  });
});