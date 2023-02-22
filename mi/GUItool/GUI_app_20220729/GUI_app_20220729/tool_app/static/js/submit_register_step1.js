/*
Submit s3_bucket path, objectives and drop_columns to Flask in step1
*/

//Ajax通信でデータ受け渡し
$(function () {

    function ajax_register(s3_output_val, objectives_val, drop_cols_val)
    {
        //メッセージ出力
        target = document.getElementById("register_block");
        target.innerHTML = "処理は数分で終わります。しばらくお待ちください。";

        //json形式へ変換
        var json_data = { "s3_output": s3_output_val, "objectives": objectives_val, "drop_cols": drop_cols_val };

        $.ajax({
            data: JSON.stringify(json_data),
            type: 'POST',
            url: '/read_s3_bucket_data/' + exp_id,
            dataType: 'json',
            contentType: 'application/json'
        }).done(function (data) {
            console.log(data.OK);
        }).always(function () {
            location.reload();
        });
    }

    function submit_register()
    {
        //処理中はボタン非活性
        btn_disable();
        //target                                          
        var s3_output_val = $('input[name="s3_output"]').val();
        var objectives_val = $('input[name="objectives"]').val();
        var drop_cols_val = $('input[name="drop_cols"]').val();
        let objectives = objectives_val.split(",");
        let pow_check = true;
        for(let i=0; i<pow_check_cols.length; i++) {
            if (objectives.includes(pow_check_cols[i]))
            {
                pow_check = false;
            }
        }
        //目的変数のオーダーが小さい場合にポップアップを出現させ、OKを押した場合に実行。NGの場合、何もしない
        if (pow_check==false)
        {
            if(confirm("目的変数のオーダーが小さいため、Shap値を出力できませんが、このまま実行してもよろしいですか？")){
                ajax_register(s3_output_val, objectives_val, drop_cols_val);
            }
            else{
                alert("ファイルアップロード前に、目的変数のスケーリングを実施してください");
                btn_able();
            }        
        }
        else
        {
            ajax_register(s3_output_val, objectives_val, drop_cols_val);
        }
    }

    //submit押下時にFlask側へデータを送信
    $('#btn_submit_register').click(function (event) {
        event.preventDefault();
        //step2,3の解析結果がある場合にポップアップを出現させ、OKを押した場合に削除。NGの場合、何もしない
        if(finished == "DONE"){
            if(confirm("解析結果が削除されますが、このままデータを登録しますか？")){
                submit_register();
            }
            else{
                btn_able();
            }
        }
        else
        {
            submit_register();
        }
        return false;
    });
})