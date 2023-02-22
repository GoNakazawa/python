/*
Submit s3_bucket path, objectives and drop_columns to Flask in step1
*/

//Ajax通信でデータ受け渡し
$(function () {

    function top_chemical()
    {
        //処理中はボタン非活性
        btn_disable();

        //target
        let top_num = $('input[name="top_num"]').val();

        //json形式へ変換
        var json_data = { "top_num": top_num };

        $.ajax({
            data: JSON.stringify(json_data),
            type: 'POST',
            url: '/draw_top_data/' + exp_id,
            dataType: 'json',
            contentType: 'application/json'
        }).done(function (data) {
            console.log(data.OK);
        }).always(function () {
            location.reload();
        });
    }

    function draw_chemical()
    {
        //処理中はボタン非活性
        btn_disable();

        //target
        let source_names = $('input[name="source_names"]').val();

        //json形式へ変換
        var json_data = { "source_names": source_names };

        $.ajax({
            data: JSON.stringify(json_data),
            type: 'POST',
            url: '/search_chemical_name/' + exp_id,
            dataType: 'json',
            contentType: 'application/json'
        }).done(function (data) {
            console.log(data.OK);
        }).always(function () {
            location.reload();
        });
    }

    //submit押下時にFlask側へデータを送信
    $('#btn_submit_top').click(function (event) {
        event.preventDefault();
        target = document.getElementById("top_block");
        target.innerHTML = "処理は数分で終わります。しばらくお待ちください。";
        //target_en = document.getElementById("top_block_en");
        //target_en.innerHTML = "Wait for a minute...";   
        top_chemical();
        return false;
    });

    $('#btn_submit_draw').click(function (event) {
        event.preventDefault();
        target = document.getElementById("draw_block");
        target.innerHTML = "処理は数秒で終わります。しばらくお待ちください。";
        //target_en = document.getElementById("draw_block_en");
        //target_en.innerHTML = "Wait for a minute...";   
        draw_chemical();
        return false;
    });
})